# topk_select Backend Tune

`aiter.topk_select` picks one of several top-k backends from the call's shape
alone. Which backend is fastest also depends on the **values**: at
`4096x8192 k=2048` on gfx950, a tie-heavy input (`equal`) runs 6.8x faster on
`sampled` than on the router's `plain` (53 µs vs 357 µs), while on `randn`
`plain` is 2.0x faster than `sampled` (46 µs vs 90 µs). This tuner measures
your shapes on your value distribution and writes a table that
`topk_select` consults before its shape rules. Validated on gfx950; the table
is keyed by `gfx` and `cu_num`, so a row only applies to the chip it was tuned
on.

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add shapes in an untuned csv, e.g. `/tmp/topk_untuned.csv`:
    |**rows**|**width**|**k**|**dist**|
    |--------|---------|-----|--------|
    |4096    |8192     |2048 |equal   |
    |512     |8192     |64   |equal   |

    Only `rows,width,k` are required. Optional columns: `dtype` (`float32`, the
    default, `bfloat16` or `float16`), `ragged` (pass `end=`), `tie` (empty,
    `low`, `high`), `deterministic`, `dist` (default `all`). `dist` is one of the
    presets `randn uniform16 bf16_randn lognormal recency equal adversarial inf`,
    `all` (every preset), a recorded `.pt` path (see
    [Recording real tensors](#recording-real-tensors)), or several of these
    joined with `;`.

    Each table row applies only to calls of the dtype it was tuned on. Today
    `topk_select` serves bfloat16 and float16 only at `k=1`, where only `argmax`
    can run, so the tuner skips such bands with "nothing to choose". When aiter
    adds a half-format backend, the same CSV tunes it.

    `aiter/configs/topk_select_untuned.csv` is not an input: as with
    `chunk_gdn_h_opt_untuned.csv`, it ships header-only and holds the band
    lookup keys that the runtime merge of tuned tables deduplicates on.

3. Start tuning:
`python3 csrc/topk_select/topk_select_tune.py -i /tmp/topk_untuned.csv -o /tmp/topk_tuned.csv`
You can find the results of this tuning in `/tmp/topk_tuned.csv`, like this
(columns after `us_default` omitted):
    |**gfx**|**cu_num**|**rows_lo**|**rows_hi**|**width_lo**|**width_hi**|**k**|**dtype**|**ragged**|**tie**|**deterministic**|**mode**|**backend**|**us**|**default_backend**|**us_default**|
    |-------|----------|-----------|-----------|------------|------------|-----|---------|----------|-------|-----------------|--------|-----------|------|-------------------|--------------|
    |gfx950 |256       |4096       |5792       |8192        |11585       |2048 |float32  |False     |       |False            |graph   |sampled    |52.9  |plain              |357.5         |

    A row covers a **band**: a half-octave range of `rows` and of `width`.
    Bands where the shape router is already the best choice get no row.

4. Verify and use the table:
`python3 csrc/topk_select/topk_select_tune.py --run_config /tmp/topk_tuned.csv`
`export AITER_CONFIG_TOPK_SELECT=/tmp/topk_tuned.csv` before the first
`topk_select` call. Without the variable, aiter reads
`aiter/configs/topk_select_tuned.csv` (ships empty) merged with
`aiter/configs/model_configs/*topk_select_tuned*.csv`; the merged result is
stored in `/tmp/aiter_configs/topk_select_tuned.csv`, as for the tuned GEMMs.
`AITER_LOG_TUNED_CONFIG=1` logs every shape that takes a table row.

## Reading the output

Each band gets one block. The runtime cannot assume exact sizes, so every
input row whose shape falls in the same band is pooled into one decision, and
each sample keeps its own exact shape.

```text
[1/2] rows 4096-5792 x width 8192-11585, k=2048, graph  (1 sample(s))
  us/call              decode     plain   sampled    stream
  equal@4096x8192      447.0     357.5*     52.9     477.1
  (* = what the router picks today)
  note: sampled beat the router in two processes on independent draws: 6.79x then 6.76x
  -> WROTE sampled: 6.76x faster than the router (geomean); worst sample equal@4096x8192 runs at 0.15x of the router's time
```

A preset sample is not one tensor. It is many independent draws of the
distribution, and the time shown is the mean per call across them: at least
`--draws` (default 16), up to 128 while they fit in 4 GiB, timed in chunks of
at most 8 GiB. Some backends' speed depends on the particular draw: `sampled`
falls back to an exact select on the rows its sample misses, and aiter measured
a call with one such row at up to 5.3x the time of the path it replaced (see
`_SAMPLED_MIN_K` in `aiter/ops/topk.py`). Serving pays the average over draws,
so that is what the tuner measures.

A row is written only if all of these hold (default `--objective no_regress`):

- the backend passes the oracle on every draw of every sample (values equal
  `torch.topk`, indices unique and in range);
- it is at most 2% slower than the router on **every** sample;
- it is at least `--min_improvement_pct` (default 3%) faster on the geometric
  mean;
- in a second, fresh process on a fresh set of draws, the same backend still
  meets both conditions above. Near-ties may swap places between the two
  measurements; the backend whose worse speedup is larger is written.

**"wrote 0 row(s)" is a normal result.** It means the shape router is already
the safest choice for what you gave it; `-o` is still written, with just the
header, so the next steps work unchanged. For example `dist=all` at
`4096x8192 k=2048` writes nothing: `sampled` wins `equal` by 6.8x but loses
`lognormal` by 2.05x, and the block says exactly that:

```text
  -> no row; the router keeps this band:
       decode: 3.58x slower than the router (plain) on lognormal@4096x8192
       sampled: 2.05x slower than the router (plain) on lognormal@4096x8192
       stream: 6.40x slower than the router (plain) on bf16_randn@4096x8192
```

If you would rather take a bounded loss on some distributions to cap the
worst case, use `--objective minimax`. It picks the backend whose worst
sample is closest to that sample's fastest backend. On the example above it
writes `sampled`.

**When a backend fails.** A candidate that returns a wrong answer, cannot be
graph-captured (in `--mode graph`), or crashes its process is dropped for
the band, and the block says which and why. If the failing backend is the
**router's own choice**, there is no baseline to beat and keeping the router
would keep the failure. The tuner then writes the fastest backend that
passed on every sample in both processes, and prints a `WARNING` naming the
failure; a wrong answer is also flagged for reporting. If no backend passed,
the band gets no row, is listed under "Failed shapes" in the summary, and the
run ends with `[Tuning not Finished]` and exit code 1, as in the other tuners.

Crashes are contained. If a band's process dies (for example a GPU memory
fault in one backend), each backend is re-run alone. Those that crash are
listed as `crashed` and excluded, and the band still gets a decision.

## Recording real tensors

Use this when you do not know your value distribution. Recording copies
device tensors to the host and synchronizes. Run it on an eager calibration
pass, never while serving. Calls inside graph capture are skipped; every
dtype `topk_select` accepts is recorded, with its dtype.

```bash
AITER_TOPK_SELECT_RECORD=/tmp/topk_rec \
AITER_TOPK_SELECT_RECORD_CALLS=4 \
AITER_TOPK_SELECT_RECORD_ROWS=64 \
  python3 your_eager_workload.py

python3 csrc/topk_select/topk_select_tune.py \
  -i /tmp/topk_rec/topk_untuned.csv -o /tmp/topk_tuned.csv --mode eager
```

`..._CALLS` caps the recordings per **band** per process, so prefill with a
different `rows` on every call still records a bounded number of files.
`..._ROWS` rows are sampled from each call and tiled back to the call's row
count when tuning. Each recorded call is one draw of your real distribution,
so record several calls per band (the default of 4 is the minimum worth
having). Ragged calls (`end=`) can only be tuned this way. The `.pt` files
are your real scores, so keep them private.

The directory is self-contained. `topk_untuned.csv` names its samples
relative to itself (`samples/...pt`), so you can record on the serving box and
tune the copied directory elsewhere. Several processes, such as the ranks of
one TP job, may record into the same directory: sample files carry the pid,
and the CSV is appended under a file lock.

Recording never fails the call it observes. If writing a sample fails
(unwritable directory, full disk), recording is switched off for the
process with one warning and `topk_select` carries on. A `..._CALLS` or
`..._ROWS` value that is not a positive integer falls back to its default,
also with one warning. Each sample file is `..._ROWS x width x 4` bytes: 64
rows of a 1M-wide row are 256 MiB.

## Verify before shipping

```bash
python3 csrc/topk_select/topk_select_tune.py --run_config /tmp/topk_tuned.csv
```

This replays every sample listed in the table through `topk_select` twice:
once with the table loaded, once with the router alone. Presets use a third
set of draws that tuning never saw.

```text
rows 2048-2896 x width 16384-23170, k=1024, eager
  sample                                table       us   router       us   ratio
  k1024_w16384_r2048_..._0.pt@2048x16384  sampled     78.0    plain    100.2   0.779
  k1024_w16384_r2048_..._1.pt@2048x16384  sampled     79.0    plain    100.2   0.789
  k1024_w16384_r2048_..._2.pt@2048x16384  sampled     79.1    plain     99.8   0.792
  k1024_w16384_r2048_..._3.pt@2048x16384  sampled     79.1    plain     99.6   0.794
  geomean: table 78.8 us vs router 99.9 us -> 0.788x
```

Each band gets a verdict in the base tuners' format (`OK` / `MISMATCH` /
`ERROR`), and the command exits 1 if any band is not `OK`. These are what the
nightly `test_run_config` checks:

| verdict | when |
|---|---|
| `MISMATCH` | the table's backend returns a wrong top-k on a sample (same oracle as tuning) |
| `ERROR` | the table's backend takes more than 1.05x the router's time on any sample (a stale row: re-tune it); or a sample could not be timed, including a recorded sample file that no longer exists; or the row lists no samples (`dists`) to check it on; or, for a fallback row (empty `us_default`), the router's choice now passes on every sample, so the row may only cost speed (re-tune it) |
| `OK` | otherwise; a fallback row is not timed against the router, and is `OK` while the router's choice still fails on some sample |

## How aiter uses the table

Only `aiter.topk_select` reads it. `top_k_per_row_prefill/decode`, and vLLM's
own top-k kernels, are not affected.

For each new call shape `(rows, width, k, dtype, ragged, tie, deterministic)`,
`topk_select` resolves a backend once and memoizes it:

1. It narrows the backends to those that can serve the shape **and** keep the
   `tie` / `deterministic` promise. The table can never widen this set.
2. It looks for table rows with this `gfx`, `cu_num`, `k`, `dtype`, `ragged`,
   `tie` and `deterministic` whose `rows` and `width` bands contain the call.
   Rows whose `mode` matches `AITER_TOPK_SELECT_TUNE_MODE` (default `graph`)
   are preferred, and the tightest band wins.
3. It uses that backend if step 1 allows it for this exact shape. Otherwise it
   logs a warning once and falls back to the shape router, as it does when
   no row matches.
4. If the table's backend raises when dispatched, that call is rerun with
   the shape router's choice, one warning is logged, and that backend is not
   used for that shape again in the process. A device-side fault
   (e.g. a memory access fault) cannot be caught this way; the tuner runs
   every candidate in its own process so such a backend is excluded before
   it reaches a table. While the tuner times a candidate this rescue is off,
   so a candidate that raises is dropped instead of being credited with the
   router's answer.

The table is resolved by `AITER_CONFIGS.AITER_CONFIG_TOPK_SELECT_FILE`, the
same `get_config_file` merge the tuned GEMMs use:

| setting | tables read |
|---|---|
| `AITER_CONFIG_TOPK_SELECT=a.csv` | `a.csv` |
| `AITER_CONFIG_TOPK_SELECT=a.csv:b.csv` | both, merged |
| unset | `aiter/configs/topk_select_tuned.csv` (ships empty) merged with every `aiter/configs/model_configs/*topk_select_tuned*.csv` |

The merge matches rows on the band lookup keys in
`aiter/configs/topk_select_untuned.csv`. If two files claim the same band, it
keeps the lower-`us` row, writes the pruned files back and raises, as for every
aiter op; `topk_select` logs one warning, serves that call with the shape
router, and reads the repaired files on the next lookup.
`op_tests/tuning_tests/test_config_shape_collision.py` catches such a clash
before it ships.

Spaces around values are ignored, so a hand-edited table loads. A file that
does not exist, a malformed row, or a row naming an unknown backend logs one
warning and is skipped. Set the variable before the first `topk_select` call,
because decisions are memoized per shape. Under graph capture the decision is
fixed at capture time. To check that the table is live: the first lookup logs
`topk_select: N tuned row(s) loaded from <file>`.

## More Options

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Append every measurement (band, pass, sample, backend, us, router, status) to this file, including dropped and crashed backends.

**Example**:
```bash
--profile_file /tmp/topk_profile.csv
```

#### `--sort`
- **Type**: Boolean (True/False)
- **Default**: `False`
- **Description**: Sort the output file by the key columns.

### Tuning Configuration

#### `--mode {graph,eager}`
- **Type**: String
- **Default**: `graph`
- **Description**: Time HIP-graph replay or eager calls; tune the way you serve. In graph mode a backend that cannot be captured (often `plain`) is dropped.

#### `--objective {no_regress,minimax}`
- **Type**: String
- **Default**: `no_regress`
- **Description**: See [Reading the output](#reading-the-output).

#### `--draws`
- **Type**: Integer
- **Default**: `16`
- **Description**: Minimum independent draws per preset sample; small shapes get up to 128 automatically.

#### `--min_improvement_pct`
- **Type**: Float
- **Default**: `3.0`
- **Description**: Geomean speedup over the router needed to write a row. With `--compare --update_improved`, also the improvement a compared row needs, as in the other tuners.

#### `--mp`
- **Type**: Integer
- **Default**: Number of available GPUs
- **Description**: Bands run in parallel, one process per band per GPU.

#### `--batch`
- **Type**: Integer
- **Default**: `100`
- **Description**: Bands per batch; the CSV is written after each batch.

#### `--all`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Re-tune bands already in `-o`. With `-i tuned.csv -o tuned.csv --all`, re-tunes a table in place (e.g. after an aiter upgrade). Without `--all`, a band already in `-o` is skipped only if its stored samples cover the requested ones; adding a distribution to a band re-tunes it. If the re-measured band now keeps the router, its earlier row is removed from `-o` (under `--compare` it is only reported, because the compare step owns the writes there).

#### `--run_config [TUNED_CSV]`
- **Type**: Optional argument
- **Default**: disabled
- **Description**: Verify a table and exit (no tuning); see [Verify before shipping](#verify-before-shipping). Exits 1 if any band is `ERROR` or `MISMATCH`.

#### `--compare`, `--update_improved`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Run the `--run_config` benchmark before and after tuning, and with `--update_improved` keep only the rows that improved by `--min_improvement_pct`.

### Profiling Configuration

#### `--warmup`
- **Type**: Integer
- **Default**: `5`
- **Description**: Untimed passes per backend over a chunk's draws before timing; the last one also sizes the timed batch (about 3 ms per round).

#### `--iters`
- **Type**: Integer
- **Default**: `5`
- **Description**: Timed rounds per backend on each chunk of draws, in shuffled order across backends; the median round is used. A round is not one call: it runs the backend over the chunk's draws (at least 4 calls) repeatedly, at least 3 times and for about 3 ms, so `--iters 5` is at least 60 timed calls per backend per chunk.

#### `--timeout`
- **Type**: Integer
- **Default**: `1800`
- **Description**: Seconds before a band's process is killed and treated as crashed.

### Debugging and Verbose Output

#### `-v, --verbose`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Also show the per-band processes' aiter logs (JIT builds, warnings).

`--errRatio`, `-k/--splitK`, `--shape_grouped` and `--e2e_tune` belong to the
GEMM tuners. This tuner ignores them and says so.

## Tuned CSV columns

```text
gfx,cu_num,rows_lo,rows_hi,width_lo,width_hi,k,dtype,ragged,tie,deterministic,mode,backend,us,default_backend,us_default,worst_ratio_vs_default,dists,n_samples,aiter_rev
```

| column | meaning |
|---|---|
| `rows_*`, `width_*` | inclusive bands with edges at `ceil(2**(e/2))` (editable) |
| `dtype` | the input dtype the row applies to |
| `us`, `us_default` | geomean µs/call of the chosen backend and of the router's choice; `us_default` is empty when the router's choice failed |
| `worst_ratio_vs_default` | chosen / router on the worst sample (empty for a router-failure row) |
| `dists` | the samples behind the decision, as `dist@ROWSxWIDTH` (recorded samples by absolute path) |
| `aiter_rev` | the git rev the table was tuned at; a mismatch at runtime logs one warning |

## Shipping a table with aiter

`aiter/configs/topk_select_tuned.csv` ships **empty on purpose**. The right
backend depends on the deployment's scores, so no single table is right for
every user. A per-model table goes in
`aiter/configs/model_configs/<model>_topk_select_tuned.csv` and is merged
automatically. Tune it on the presets that match that model's scores,
because its `dists` must name presets: a recorded sample path exists only on
the machine that recorded it, so CI could never re-verify the row, and
`test_csv_validation` rejects it. The tuning test suite covers such a table:

```bash
python3 -m unittest op_tests.tuning_tests.test_csv_validation   # format, duplicates, backends
python3 -m unittest op_tests.tuning_tests.test_config_shape_collision.TestConfigShapeCollision.test_topk_select
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfig.test_topk_select
python3 -m unittest op_tests.tuning_tests.test_topk_select_tuner  # this tuner's own tests
python3 -m pytest op_tests/tuning_tests/test_tune_pipeline.py -k topk_select
```

## Adding a backend to topk_select

Register it in `aiter/ops/topk_select.py` the way the existing ones are, and
the tuner needs no change:

- `_available(width, k, wave_size, ragged, fp32, sampled_ok)`: where it can
  run at all (and for which dtypes, through `fp32`);
- `_ROW_PREDICATES`: any limit that also depends on the row count or the GPU,
  as `sampled`'s does. A table may only pick a backend from `_servable()`,
  which applies these;
- `_BACKENDS_BY_TIE` and `_NONDETERMINISTIC`: which `tie` / `deterministic`
  promises it keeps;
- `_dispatch`: how to call it.

The tuner asks `_servable()` and the router for candidates, times every
backend through `topk_select` itself, and the table loader and
`test_csv_validation` read the backend names from `_BACKENDS_BY_TIE`.
Existing tables keep working; re-tune them with
`-i tuned.csv -o tuned.csv --all` to let the new backend compete. A table
naming a backend that was later removed or renamed logs one warning and the
row is ignored.

## Notes

- Half formats: supported in the table and the tuner, but `topk_select`
  serves bf16/fp16 only at `k=1` with `argmax` alone, so there is nothing
  to choose yet.
- Ragged shapes only from recordings; there are no ragged presets.
- Backend parameters (e.g. `sampled`'s sample size, `stream`'s split) are
  not tuned, only the backend choice.
- Validated on gfx950 (cu_num 256). The code has no gfx950 assumption, but
  no other chip has been measured.
- No online tuning or drift detection. Re-tune when the score distribution
  or aiter changes. `--run_config` in CI reports `ERROR` on a row that has
  gone stale.
