# FlyDSL all-reduce tuning

Turn-key tooling that measures the FlyDSL all-reduce kernels on the machine at hand and fits the
dispatch tables that decide, per message size, *which* kernel runs. One driver script, `tune.sh`,
runs every measurement and fit; the only manual step is pasting the fitted values into
`aiter/ops/flydsl/allreduce_policy.py` and copying the fitted fused ladder file over
`aiter/ops/flydsl/fused_allreduce_ladders.json` (step 5).

Every measurement is the `bench` scenario of the two multi-GPU tests, so tuning and the tests time
kernels the same way (`flydsl_allreduce_bench.py` describes the method):

* `op_tests/multigpu_tests/test_flydsl_one_shot_allreduce.py` -- the one-shot, against
  `CustomAllreduce` and RCCL;
* `op_tests/multigpu_tests/test_flydsl_quick_allreduce.py` -- the mesh and the ring, against the HIP
  quick all-reduce.

Both take `--operation ar` (the plain all-reduce) or `--operation ar_rmsnorm` (all-reduce + residual
add + RMSNorm), and `--candidates candidates.csv --candidate-set <set>` to time pinned FlyDSL
configurations next to the shipping engines.

Two independent tuning problems share the tooling:

| Problem | What is tuned | Tables it feeds (`allreduce_policy.py`) |
|---|---|---|
| **Plain all-reduce** | which family (one-shot / quantized mesh / quantized ring) per payload size, and which instance (block, super-tile, grid cap) inside each family | `FAMILY_POLICIES`, `ONESHOT_LADDER`, `MESH_ST_LADDER`, `RING_ST_LADDER` |
| **Fused all-reduce + RMSNorm** | one-shot rungs (atoms, grid cap, hidden-dim split), the quantized mesh/ring instance (atoms per row, super-tile), each as a default ladder and per hidden size, and the family windows | `FUSED_FAMILY_POLICIES`; the ladder file `fused_allreduce_ladders.json`, read into `FUSED_ONESHOT_LADDER`, `FUSED_MESH_ST_LADDER`, `FUSED_RING_ST_LADDER`, `FUSED_QR_ROW_ATOMS` |

Tables are keyed `(arch, link, world_size)` with `link` in `pcie` / `xgmi`; the fused ladders also by the hidden
size (see [Fused ladder file](#fused-ladder-file)). The driver detects the
fabric from the KFD topology (`allreduce_policy.detect_link()`), and the fits read the arch from the
CSVs, so a run on an 8x MI355X box fits the `("gfx950", "xgmi", N)` rows. Override the fabric with
`--link`.

## Quick start

```bash
cd op_tests/multigpu_tests/flydsl_allreduce_tuning
./tune.sh setup            # once per host: flydsl pin, pyzmq, tabulate
./tune.sh --smoke all      # a few minutes of plumbing check: tiny sweep + fit, results are NOT meaningful
./tune.sh all              # the real thing: check, sweep, fit, fused-sweep, fused-fit
# -> paste fitted values into allreduce_policy.py and copy out/<host>/fused/fused_allreduce_ladders.json
#    over aiter/ops/flydsl/fused_allreduce_ladders.json (step 5), then
./tune.sh test && ./tune.sh validate && ./tune.sh audit
```

`tune.sh --help` lists every command and option. Output goes to `out/<hostname>/` next to the
script (git-ignored); change it with `--out DIR`.

## Running under Slurm

`slurm_tune.sbatch` runs `tune.sh` in the `rocm/pytorch` Docker image on one whole node (8 GPUs):
it clones the branch on the host, installs the Python deps in the container, runs `--smoke all` as a
gate, then `all`. Results go to `~/flydsl_tuning/job-<id>/out`; header of the script lists the knobs.

```bash
DRY_RUN=1 bash slurm_tune.sbatch    # no GPU needed: image, clone, deps, `tune.sh --dry-run all`
sbatch slurm_tune.sbatch            # the real run; after a time-out: RUN_NAME=job-<id> sbatch ...
```

`tune.sh --dry-run <cmd>` is the same rehearsal without Docker: it does not require GPUs, lists every
chunk with its bench command, checks the input files exist, and only loads the fit scripts.

## Prerequisites

* A **quiet** machine: all GPUs of the world sizes you tune must be idle. Other jobs on the GPUs
  (or heavy CPU load) show up as timing noise and bias the fit. Sweeps for world sizes larger than
  the visible GPU count are skipped automatically.
* The repo's Python environment (torch + ROCm, `aiter` importable from this checkout).
  `tune.sh` sets `PYTHONPATH`, `AITER_META_DIR` and the FlyDSL cache variable itself; no docker is needed.
* `./tune.sh setup` installs what is usually missing: the `flydsl` version pinned in
  `requirements.txt`, `pyzmq` and `tabulate`. `./tune.sh check` reports anything still wrong
  (GPU count, imports, flydsl version, fabric type, whether the tests load, whether
  `candidates.csv` is current).

Why the environment variables matter (all set by `tune.sh`, listed so you do not have to rediscover them):

* `AITER_META_DIR=<repo root>` + `PYTHONPATH=<repo root>`: without them an install-mode aiter picks up a
  stale, git-ignored `aiter_meta/` build artifact and `import aiter` fails (`get_kernel_instance`).
* `FLYDSL_EXTRA_SOURCE_DIRS=<repo>/aiter/ops/flydsl/kernels{,/comm}` **is critical for validation**:
  FlyDSL's disk cache does not hash the shared kernel modules. Without it, after you edit a kernel the
  benchmark silently runs stale compiled kernels and the audit "passes" against the old code.

## Candidates

`candidates.csv` lists the pinned FlyDSL configurations the sweeps time, one per row: a name, the
operation (`ar`/`ar_rmsnorm`), the algorithm (`oneshot`/`mesh`/`ring`), one column per knob (empty:
the engine's own) and the sets it belongs to. It is generated by `gen_candidates.py` -- the fused grids
depend on which row geometries the kernels can build -- and checked in; edit the generator and re-run it.

| Set | Rows | Used by |
|---|---|---|
| `sweep` | plain one-shot, mesh and ring grids | `sweep` |
| `validate` | plain one-shot and mesh grids (the choices the shipped dispatcher has) | `validate` |
| `fused` | fused one-shot: every legal block x grid cap, and every legal split x block x cap | `fused-sweep` (one-shot chunks) |
| `fused_qr` | fused mesh/ring: atoms per row x super-tile x grid cap, and the low-latency mesh build (`fused_fly_mesh_ll_a<atoms_per_row>_st1_g<cap>`) | `fused-sweep` (`qr_` chunks) |
| `smoke` | a few rows of each | `--smoke` |

A row is timed only where it builds exactly as named; elsewhere its cell is empty (n/a): a block or
split the width does not have, an `atoms_per_row` the width or world size does not have, a fused grid
cap above the CU count (a fused launch is held to one workgroup per CU; two for the low-latency build), a
low-latency row at a width whose block is not 256 threads (`hidden != 2048 * atoms_per_row`), or a one-shot
payload above `AITER_BENCH_FLY1S_MAX_KB` (default 64 MiB).

To see every configuration a test can pin -- not only the tuning grid -- run it with `--list-candidates`. It
prints a candidate file on stdout, for `--operation` (default `ar`), the quick test's `-a` schedules and, for
the fused rows, the widths and world sizes of the run (`-s`/`--shape-csv`, `--tp`; default the bench's).
Edit or filter it, then time it like `candidates.csv`:

```bash
python ../test_flydsl_quick_allreduce.py --list-candidates --operation ar_rmsnorm -a mesh --shape-csv fused/shapes/fused_w4096.csv > mine.csv
python ../test_flydsl_quick_allreduce.py --scenario bench --operation ar_rmsnorm -a mesh --tp 8 --shape-csv fused/shapes/fused_w4096.csv --candidates mine.csv
```

The names are the column prefixes of the bench tables and the keys the fits report. The shipping
engines and stock kernels are read under fixed keys (`bench_io.SHIPPING`): `fly_1stage`, `fly_int4`,
`fly_int4_ring`, `qr_int4`, `cdr`, `cdr_naive`, `rccl` for the plain bench, and `fused_fly_1stage`,
`fused_fly_mesh`, `fused_fly_ring`, `fused_cdr_{1,2}stage`, `separate_{cdr,rccl,qr_int4}`, `fused_qr_int4`
for the fused one.

Of the stock kernels, the tuning chunks (`sweep`, `validate`, `fused-sweep`) time only the reference that the
policy windows are placed against (the tests' `--baselines reference`): what production runs where FlyDSL
declines a payload. In the one-shot test that is `CustomAllreduce`'s `cross_device_reduce` (`cdr`), or its
fused kernel at both stage counts (`cdr_fused_{1,2}stage`); in the quick test the HIP quick all-reduce at
INT4 (`hip`), or its fused kernel (`hip_fused`). `cdr_naive`, RCCL, the two-launch `<all-reduce>+norm`
paths and the unfused FlyDSL paths are then neither built nor timed, which saves the slowest columns at large
payloads. `--all-baselines` times them all again; `report` always does.

## Step-by-step

Every sweep step is resumable: a finished chunk (its CSV exists) is skipped on re-run, and a failed
chunk is reported at the end and retried by simply running the command again. `--force` redoes everything.

### 1. Check the environment

```bash
./tune.sh check
```

### 2. Plain all-reduce sweep

```bash
./tune.sh sweep                 # all of --tp "8 4 2"
./tune.sh --tp "8" sweep        # just TP8
```

For each world size and each shape set, runs both tests' bench with `--candidate-set sweep`: the
one-shot test times the shipping one-shot, every pinned one-shot row and `cdr` (with `--all-baselines`,
`cdr_naive` and RCCL too); the quick test times the shipping mesh and ring, every pinned mesh and ring row
and the HIP quick all-reduce at INT4. Outputs `out/<host>/sweep/{oneshot,quick}_tp<N>_<shapes>.{csv,md,log}`.

Shape sets (`shapes/*.csv`, columns `M,K,label`):

| File | Purpose |
|---|---|
| `ar_sweep_a_small.csv` | decode-sized messages (14 KiB ... 450 KiB), K=7168 |
| `ar_sweep_a_large.csv` | prefill-sized messages (4 MiB ... 112 MiB), K=7168 |
| `ar_sweep_b_ksens.csv` | K-sensitivity holdout: same payload sizes at K=4096/7168/8192. Never fitted; it checks that payload bytes is a valid dispatch key |

### 3. Fit the plain tables

```bash
./tune.sh fit
```

Runs `fit_allreduce_policy.py` on the `a_small`+`a_large` sweeps (both tests' CSVs, joined on the
shape) with `b_ksens` as holdout, and writes `out/<host>/fit/fit.txt`. The fit searches round
thresholds (`2^k` and `1.5*2^k`) minimising worst-case regret against a per-shape oracle. Thresholds
are half-open, as `Window` and `select_rung` are: below a threshold one side, from it the other. Read
the report top to bottom:

* `fast` row: the one-shot ceiling (one-shot -> quantized mesh) and the mesh ceiling (mesh -> ring).
* `holdout K=...` lines: must be `worst 1.000x`-ish. If a holdout K is clearly worse, payload bytes is
  the wrong key for this fabric and the tables below it are not trustworthy -- stop and investigate.
* `exact vs primary/robust`: the exact one-shot ceiling, where aiter `cdr` overtakes the one-shot
  (what exact mode falls through to). `robust` adds `cdr_naive` and RCCL as a cross-check, so it is
  only there for a sweep run with `--all-baselines`.
* `ladder oneshot/mesh/ring`: the instance per size range (`[min_bytes: candidate]`), with the worst regret
  and how it improves with more rungs. `ladder ring -- not dispatched here` means the ring never won
  and the existing rows can stay.
* The `## paste-ready` block at the end holds the `FamilyPolicy(...)` rows of `FAMILY_POLICIES` (checked
  by building them) and the `ONESHOT_LADDER`, `MESH_ST_LADDER` and `RING_ST_LADDER` rows as rung
  literals.

### 4. Fused all-reduce + RMSNorm sweep and fit

```bash
./tune.sh fused-sweep                          # TP 8 4 2 x widths 2048 3072 4096 7168 8192
./tune.sh --tp "8 4" --widths "7168" fused-sweep   # a subset
./tune.sh fused-fit
```

The sweep runs both tests' bench with `--operation ar_rmsnorm`, two chunks per (TP, width), for
M = 1..2048 at each width (`fused/shapes/fused_w*.csv`):

* `tp<N>_w<H>`: the one-shot test with `--candidate-set fused`: the shipping fused one-shot, every pinned
  fused one-shot row, and `CustomAllreduce`'s fused kernel (1-stage and 2-stage) (with `--all-baselines`,
  also the two-launch `cdr`/RCCL + RMSNorm).
* `qr_tp<N>_w<H>`: the quick test with `--candidate-set fused_qr`: the shipping fused mesh and ring, every
  pinned fused mesh/ring row, the low-latency mesh rows among them, and the HIP quick all-reduce's fused
  kernel at INT4 where its row constraint allows (with `--all-baselines`, also its two-launch form,
  `CustomAllreduce`'s kernels and the unfused FlyDSL paths). Being its own chunk, an older sweep directory
  picks it up on a plain re-run of `fused-sweep` without redoing the one-shot.

Output: `out/<host>/fused/sweep/{,qr_}tp<N>_w<H>.{csv,md,log}`.

`fused-fit` writes `out/<host>/fused/fit.txt` (and a per-shape comparison in `fused/summary.txt`) and the
ladder file `out/<host>/fused/fused_allreduce_ladders.json`. Per world size it prints:

* the one-shot ladders, found by dynamic programming over payload breakpoints (`--rungs N`, default 3,
  penalised per rung), as ladder-file rungs:
  * `oneshot default -- every width`: one ladder fitted over all swept widths, with `split=1` only. It runs at
    every width without a ladder of its own, including widths that were never swept.
  * `oneshot hidden=H`: a ladder fitted on width H alone, splits included. A rung's `split` resolves to the
    nearest split the width has. The width keeps it only when it beats the default there by `--width-gain`
    (geomean, default 2%); `default/own` is that speedup. Otherwise the line says the width runs the default.
* `mesh` / `ring`: the same, default and per width, for the quantized schedules: an `atoms_per_row` and a
  ladder of `(min_bytes, super_tile, grid_cap)` rungs, super-tile and cap chosen together per rung. On xGMI
  a mesh rung of a width's own ladder may be the low-latency build (`"low_latency": true`), where its rows
  won; it exists only at the width whose row is a 256-thread block at that `atoms_per_row`, so the default
  ladder never takes it. The
  `atoms_per_row` is the outer choice -- the line `log-regret by atoms_per_row` shows the margin -- and each
  candidate gets its own super-tile ladder; a width without that `atoms_per_row` runs the nearest it has, as the
  engine does. `shipped/fitted` is the speedup over the ladders the policy reads now (default or per width)
  inside that family's window. Fitted above the quantized one-shot ceiling only, since nothing below it
  dispatches to the mesh or ring; a family no swept shape dispatches to is "not dispatched" and keeps its
  ladders. Needs the `qr_` chunks; without them the fit says so and fits the one-shot only.
* the exact one-shot ceiling (vs cdr), the quantized one-shot ceiling (vs the fitted mesh/ring), the
  mesh ceiling (vs ring) and the quantized ceiling (vs the fallback), combined into a paste-ready
  `FamilyPolicy(...)` row of `FUSED_FAMILY_POLICIES`. The fallback is what production runs with
  quantization on where FlyDSL declines: the HIP quick all-reduce's fused kernel at INT4 (`hip_fused`)
  where `CudaCommunicator` dispatches it -- not at its `use_1stage` sizes, nor for rows that do not divide
  its 32 KiB tile, nor below its size floor -- and `CustomAllreduce`'s fused kernel elsewhere. It also
  scores the gap between the exact one-shot window and the start of the quantized ones. The windows are
  fitted against what each width runs (its own ladder, or the default). `mesh=None` means the ring won
  everywhere above the one-shot; an open-ended last window that FlyDSL won up to the top of the sweep.
* a per-shape table and the in-window geomeans: `ladder/best` (how close the fitted ladders are to the
  per-shape best), `default/ladder` (what the per-width ladders gain over the default), `cdr/ladder` (the
  speedup over aiter) and `shipped/ladder` (how much the currently shipped ladders lose against the fitted
  ones).

The ladder file is the base file (`--base`; by default the file the policy reads, see below) with the fitted
families of the swept platforms replaced: the default ladder, and for each swept width its own ladder, or
none. A width the sweep did not cover keeps its entry, and the fit says so. Platforms not swept are unchanged.

### Fused ladder file

The fused ladders are data, not code: `aiter/ops/flydsl/fused_allreduce_ladders.json`, read when
`allreduce_policy` is imported (the format is described there, under "Level 2: configuration ladders").
Per platform and family it has a `default` ladder and, under `hidden`, ladders for single widths:

```json
"oneshot": {
  "default": {"rungs": [{"min_bytes": 0, "atoms": 1, "grid_cap": 128, "fanout": "peer", "split": 1}]},
  "hidden": {"7168": {"rungs": [
    {"min_bytes": 0, "atoms": 1, "grid_cap": 128, "fanout": "peer", "split": 8},
    {"min_bytes": "48 KiB", "atoms": 1, "grid_cap": 32, "fanout": "peer", "split": 1}]}}
},
"mesh": {"default": {"atoms_per_row": 1, "rungs": [{"min_bytes": 0, "super_tile": 8, "grid_cap": 32}]}}
```

* A width with no entry runs the default. The default one-shot ladder never splits (`split=1`), and loading a
  file whose default splits fails.
* A mesh/ring width entry without `atoms_per_row` takes the default's; `null` is the widest block.
* `AITER_FLY_AR_FUSED_LADDERS=<file>` makes the policy read another file instead -- every rank must point at
  the same one, or the fused path declines. That is how a fitted file is tried before it is copied into the
  tree (step 6).
* The `7168` entries shipped today are placeholders: the ladders shipped before the file existed, which were
  measured at that width. Until the widths are tuned, the default is the same ladders unsplit.

### 5. Paste the fitted values into `allreduce_policy.py` and copy the ladder file (manual)

| Fit output | Table in `aiter/ops/flydsl/allreduce_policy.py` |
|---|---|
| `FamilyPolicy(...)` rows of `FAMILY_POLICIES` | `FAMILY_POLICIES` |
| `ONESHOT_LADDER` / `MESH_ST_LADDER` / `RING_ST_LADDER` rows | the table of the same name (leave the ring unchanged if it is "not dispatched here") |
| fused `fused_allreduce_ladders.json` (one-shot, mesh and ring ladders, default and per width, with `atoms_per_row`) | copy over `aiter/ops/flydsl/fused_allreduce_ladders.json` |
| fused `FUSED_FAMILY_POLICIES row` | `FUSED_FAMILY_POLICIES` |

Things the fit does **not** decide, which you should set by hand from the sweep tables (`sweep/*.md`):

* A floor below which the one-shot declines and aiter `cdr` is used instead: the `oneshot` window's `min`.
  Take it from the smallest size at which the one-shot beats `cdr` stably. You can try a value without editing
  the source via `AITER_FLY_AR_ONESHOT_MIN_BYTES` (`AITER_FLY_AR_FUSED_ONESHOT_MIN_BYTES` for the fused path).
* The emitted `oneshot` window is the exact one-shot ceiling (vs `cdr`); the quantized ceiling (vs the
  mesh/ring) only places the start of the mesh or the ring. The one-shot window may extend into the mesh/ring
  windows, which take their payloads first with quantization on, or end below the mesh, leaving a gap that
  falls through to `cdr`; `FamilyPolicy` rejects only a mesh and a ring window that overlap.
* Fitted values are exact at the sweep's resolution (about 4 points per octave); do not read more precision
  into them than the round numbers say.

After editing, run the unit tests that check the table invariants (needs the environment `tune.sh` sets up,
hence the wrapper):

```bash
./tune.sh test
```

### 6. Validate and audit the pasted tables

```bash
./tune.sh validate     # re-runs the shipping engines (now on the edited tables) next to the pinned rows
./tune.sh audit        # PASS/FAIL: the shipped dispatcher vs the best pinned row at every shape (10% tolerance)
```

`validate` runs both tests at all shapes, including the holdout, with `--candidate-set validate`. `audit`
reconstructs the shipped dispatcher (`fly_auto`) at every shape -- the shipping engine of the family the
edited policy routes the payload to -- in both accuracy regimes (`fast`: quick reduce at INT4, every family;
`exact`: the exact one-shot only, declining to `cdr` above its window), and fails if it is more than 10%
slower than the best pinned candidate at any shape; a miss against an *identical* kernel is reported as
"same-kernel noise" and does not fail. In `exact` mode the declined sizes are audited too: a one-shot that
beats `cdr` there by more than the tolerance is a ceiling that should be wider. A FAIL names the shape and the
pinned instance that beat the dispatcher -- fix the table row covering that size and re-run
`validate --force` for the affected world size (`--tp N`).

Do not skip the re-sweep: the shipping engines read the tables when they are built, so auditing the step-2
sweep only measures the *old* tables.

For the fused tables there is no separate validate command: re-run the fused sweep into a fresh directory
and check that the in-window `shipped/ladder` geomean and each mesh/ring `shipped/fitted` are ~1.00 (the
shipped tables now equal the fitted ones), and that `auto` is not losing to the best pinned row in
`fused/summary.txt`. The fitted ladder file can be tried before it is copied into the tree, through
`AITER_FLY_AR_FUSED_LADDERS` (an absolute path: the benchmarks run from other directories):

```bash
export AITER_FLY_AR_FUSED_LADDERS=$PWD/out/$(hostname -s)/fused/fused_allreduce_ladders.json
./tune.sh test                                   # the table invariants, on the fitted file
./tune.sh --out out/after fused-sweep && ./tune.sh --out out/after fused-fit
unset AITER_FLY_AR_FUSED_LADDERS
```

### 7. Optional: reference report

```bash
./tune.sh report
```

Writes the tests' own bench tables -- the shipping engines against the stock kernels over each test's
default shapes, for both operations, per world size -- to `out/<host>/report/`: the form used for
performance write-ups.

## Known pitfalls

* **`cdr` is bimodal on xGMI**: its 1-stage/2-stage choice can flip between runs, so a single pass can score
  the wrong baseline. Compare across passes before trusting a small win or loss against `cdr`.
* Run-to-run noise: very small payloads (<20 us) are sensitive to host load. Each candidate's time is the
  slowest rank's median over rounds whose order rotates (see `flydsl_allreduce_bench.py`).
* Adding a kernel knob or candidate: add the knob to `CANDIDATE_KNOBS` in `op_tests/multigpu_tests/flydsl_allreduce_bench.py` and
  to the engine construction in the test, add rows in `gen_candidates.py`, and re-run it.
* Everything here is read-only with respect to the source tree: nothing but you edits the policy tables.

## Layout

```
tune.sh                     driver (setup/check/sweep/fit/fused-*/validate/audit/report)
candidates.csv              pinned configurations the sweeps time (generated)
gen_candidates.py           writes candidates.csv
bench_io.py                 reads the tests' bench CSVs into the fits' columns
fit_allreduce_policy.py     plain fit + --audit-auto
shapes/                     plain sweep shape sets (M,K,label)
fused/analyze_fused.py      per-shape summary of the fused sweep
fused/fit_fused.py          fused ladders (one-shot, mesh, ring; default and per width), row width and window fit
fused/ladders_json.py       writes the fused ladder file (fused_allreduce_ladders.json) the fit emits
fused/shapes/               fused sweep shape sets, one per hidden size
out/                        sweep data, fits, logs (git-ignored; large)
```
