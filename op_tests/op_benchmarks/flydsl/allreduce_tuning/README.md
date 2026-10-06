# FlyDSL all-reduce tuning

Turn-key tooling that measures the FlyDSL all-reduce kernels on the machine at hand and fits the
dispatch tables that decide, per message size, *which* kernel runs. One driver script, `tune.sh`,
runs every measurement and fit; the only manual step is pasting the fitted values into the source
tables (step 5).

Two independent tuning problems share the tooling:

| Problem | What is tuned | Tables it feeds |
|---|---|---|
| **Plain all-reduce** | which family (one-shot / quantized mesh / quantized ring) per payload size, and which instance (block, super-tile, grid cap) inside each family | `FAMILY_POLICY`, `ONESHOT_LADDER`, `MESH_ST_LADDER`, `RING_ST_LADDER` |
| **Fused all-reduce + RMSNorm** | one-shot rungs (atoms, grid cap, hidden-dim split) and the one-shot windows | `FUSED_ONESHOT_LADDER`, `FUSED_FAMILY_POLICY` |

Tables are keyed `(link, world_size)` with `link` in `pcie` / `xgmi`. The driver detects the fabric
from the KFD topology (`has_xgmi_peer_links()`), so a run on an 8x MI325X box fits the `("xgmi", N)`
rows and a run on a PCIe-only box fits the `("pcie", N)` rows. Override with `--link`.

## Quick start

```bash
cd op_tests/op_benchmarks/flydsl/allreduce_tuning
./tune.sh setup            # once per host: flydsl pin, pyzmq, tabulate
./tune.sh --smoke all      # ~3 min plumbing check: tiny sweep + fit, results are NOT meaningful
./tune.sh all              # the real thing: check, sweep, fit, fused-sweep, fused-fit (~1.5 h on 8 GPUs)
# -> paste fitted values into the source tables (step 5), then
./tune.sh test && ./tune.sh validate && ./tune.sh audit
```

`tune.sh --help` lists every command and option. Output goes to `out/<hostname>/` next to the
script (git-ignored); change it with `--out DIR`.

## Prerequisites

* A **quiet** machine: all GPUs of the world sizes you tune must be idle. Other jobs on the GPUs
  (or heavy CPU load) show up as timing noise and bias the fit. Sweeps for world sizes larger than
  the visible GPU count are skipped automatically.
* The repo's Python environment (torch + ROCm, `aiter` importable from this checkout).
  `tune.sh` sets `PYTHONPATH`, `AITER_META_DIR` and the FlyDSL cache variable itself; no docker is needed.
* `./tune.sh setup` installs what is usually missing: the `flydsl` version pinned in
  `requirements.txt`, `pyzmq` and `tabulate`. `./tune.sh check` reports anything still wrong
  (GPU count, imports, flydsl version, fabric type, whether the benchmark loads).

Why the environment variables matter (all set by `tune.sh`, listed so you do not have to rediscover them):

* `AITER_FLY_AR=1` opts the FlyDSL all-reduce path in.
* `AITER_META_DIR=<repo root>` + `PYTHONPATH=<repo root>`: without them an install-mode aiter picks up a
  stale, git-ignored `aiter_meta/` build artifact and `import aiter` fails (`get_kernel_instance`).
* `FLYDSL_EXTRA_SOURCE_DIRS=<repo>/aiter/ops/flydsl/kernels` **is critical for validation**: FlyDSL's
  disk cache does not hash the shared kernel modules where the ladder tables live. Without it, after
  you edit a table the benchmark silently runs stale compiled kernels and the audit "passes" against
  the old tables.

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

For each world size and each shape set, runs `bench_comm.py` (`--operation ar`) (`--timing graph`, 5 warmup /
50 iterations, `--fly-accuracy fast`) over ~40 candidates: aiter references (`cdr`, `cdr_naive`,
`qr_int4`), the shipped dispatcher (`fly_auto`, `fly_1stage`) and every pinned FlyDSL instance
(one-shot, mesh, ring). Outputs `out/<host>/sweep/tp<N>_fast_<shapes>.{csv,md,log}`.
About 30 minutes for TP 2/4/8.

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

Runs `fit_allreduce_policy.py` on the `a_small`+`a_large` sweeps with `b_ksens` as holdout, and writes
`out/<host>/fit/fit.txt`. The fit searches round thresholds (`2^k` and `1.5*2^k`) minimising worst-case
regret against a per-shape oracle. Read the report top to bottom:

* `fast` row: thresholds for `oneshot_max` (one-shot -> quantized mesh) and `mesh_max` (mesh -> ring).
* `holdout K=...` lines: must be `worst 1.000x`-ish. If a holdout K is clearly worse, payload bytes is
  the wrong key for this fabric and the tables below it are not trustworthy -- stop and investigate.
* `exact vs primary/robust`: `oneshot_max_exact`, where aiter `cdr`/RCCL overtakes the one-shot
  (what exact mode falls through to).
* `ladder oneshot/mesh/ring`: the instance per size range (`[min_bytes: candidate]`), with the worst regret
  and how it improves with more rungs. `ladder ring -- not dispatched here` means the ring never won
  and the existing rows can stay.
* The `## paste-ready` block at the end holds Python literals for `FAMILY_POLICY` (as `_FAMILY_POLICY`)
  and `ONESHOT_LADDER` (as `_ONESHOT_LADDER`). The mesh/ring ladders are printed as comment lines with
  candidate names; translate them as described in step 5.

### 4. Fused all-reduce + RMSNorm sweep and fit

```bash
./tune.sh fused-sweep                          # TP 8 4 2 x widths 3072 4096 7168 8192
./tune.sh --tp "8 4" --widths "7168" fused-sweep   # a subset
./tune.sh fused-fit
```

The sweep (`--fusion ar_rmsnorm`) runs every fused one-shot row -- ladder default, pinned unsplit grids, and
all legal hidden-dim split (`k`) rows (137 rows) -- plus the `cdr` fused/separate baselines, `fused_fly_auto`
and the quantized mesh, for M = 1..2048 at each width (`fused/shapes/fused_w*.csv`). About 45 minutes for the
full set; output `out/<host>/fused/sweep/tp<N>_w<H>.{csv,md,log}`.

`fused-fit` writes `out/<host>/fused/fit.txt` (and a per-shape comparison in `fused/summary.txt`). Per world
size it prints:

* `FUSED_ONESHOT_LADDER rungs`: paste-ready rungs `(min_bytes, atoms, grid_cap, "peer", split)`,
  found by dynamic programming over payload breakpoints (`--rungs N`, default 3, penalised per rung).
* `oneshot_max_exact (vs cdr)` and `oneshot_max (vs mesh)`: the `FUSED_FAMILY_POLICY` bounds.
* a per-shape table and the in-window geomeans: `ladder/best` (how close the fitted ladder is to the
  per-shape best), `cdr/ladder` (the speedup over aiter) and `shipped/ladder` (how much the currently
  shipped ladder loses against the fitted one).

Rungs are shared by all widths; a rung's `split` resolves per width to the nearest legal split, so check that
widths with few legal splits (e.g. 3072, only k=2) do not show large regret.

### 5. Paste the fitted values into the source tables (manual)

| Fit output | Source table (rows `("xgmi", N)` or `("pcie", N)`) |
|---|---|
| `_FAMILY_POLICY` (`oneshot_max`, `oneshot_max_exact`, `mesh_max`) | `FAMILY_POLICY` in `aiter/ops/flydsl/allreduce_policy.py` |
| `_ONESHOT_LADDER` | `ONESHOT_LADDER` in `aiter/ops/flydsl/kernels/one_shot_allreduce.py` |
| `ladder mesh` comment lines | `MESH_ST_LADDER` in `aiter/ops/flydsl/kernels/quick_allreduce_mesh.py` |
| `ladder ring` comment lines | `RING_ST_LADDER` in `aiter/ops/flydsl/kernels/quick_allreduce_ring.py` (leave unchanged if ring is "not dispatched here") |
| fused `FUSED_ONESHOT_LADDER rungs` | `FUSED_ONESHOT_LADDER` in `aiter/ops/flydsl/kernels/one_shot_allreduce.py` |
| fused `oneshot_max` / `oneshot_max_exact` | `FUSED_FAMILY_POLICY` in `aiter/ops/flydsl/allreduce_policy.py` (keep `mesh_max=None` where the ring never wins) |

Translating candidate names from the mesh/ring ladder lines into rungs
`(min_bytes, super_tile, grid_cap, block)`:

* mesh `fly_int4_b256_st8_g128` -> `(min_bytes, 8, 128, 256)` (`b`=block, `st`=super-tile,
  `g`=grid cap)
* ring `fly_int4_ring_b512_st16_g128` -> `(min_bytes, 16, 128, 512)`
* `min_bytes` is the byte count printed in front of each candidate (first rung is 0).

Things the fit does **not** decide, which you should set by hand from the sweep tables (`sweep/*.md`):

* `FamilyPolicy.min_bytes`, the floor below which the custom-AR slot declines and aiter `cdr` is used instead.
  Take it from the smallest size at which `fly_1stage` beats `cdr` stably. You can try a value without editing
  the source via `AITER_FLY_AR_ONESHOT_MIN_BYTES`. (`AITER_CUSTOM_AR_MIN_SIZE` is the wrong lever: it sends
  sub-floor messages to RCCL, not `cdr`.)
* A paste-ready line that says `oneshot_max_exact (...) fitted below oneshot_max (...)` is an advisory;
  the printed `FamilyPolicy` already widens it. `oneshot_max_exact` may be smaller than `oneshot_max` -- the
  two ceilings are independent.
* Fitted values are exact at the sweep's resolution (about 4 points per octave); do not read more precision
  into them than the round numbers say.

After editing, run the unit tests that check the table invariants (needs the environment `tune.sh` sets up,
hence the wrapper):

```bash
./tune.sh test
```

### 6. Validate and audit the pasted tables

```bash
./tune.sh validate     # ~30+ min: re-runs fly_auto (now using the edited tables) next to every pinned row
./tune.sh audit        # PASS/FAIL: fly_auto vs the best pinned row at every shape (10% tolerance)
```

`validate` runs both accuracy regimes (`fast` and `exact`) at all shapes, including the holdout. `audit`
fails if `fly_auto` is more than 10% slower than the best pinned candidate at any shape; a miss against an
*identical* kernel is reported as "same-kernel noise" and does not fail. A FAIL names the shape and the
pinned instance that beat the dispatcher -- fix the table row covering that size and re-run `validate --force`
for the affected world size (`--tp N`).

Do not skip the re-sweep: `--audit-auto` grades the `fly_auto` column that is stored in the CSV, so auditing the
step-2 sweep only measures the *old* tables. This is also why `FLYDSL_EXTRA_SOURCE_DIRS` has to be set (the
driver does it).

For the fused tables there is no separate validate command: re-run the fused sweep into a fresh directory
(`./tune.sh --out out/after fused-sweep && ./tune.sh --out out/after fused-fit`) and check that the in-window
`shipped/ladder` geomean is ~1.00 (the shipped ladder now equals the fitted one) and `fused_fly_auto` is
not losing to the best pinned row in `fused/summary.txt`.

### 7. Optional: reference report

```bash
./tune.sh report
```

Writes markdown tables of `fly_auto` against `cdr`, `cdr_naive`, `qr_int4`, `rccl` over the default shape list
(fast and exact regime, per world size) to `out/<host>/report/` -- the form used for performance write-ups.

## Known pitfalls

* **`cdr` is bimodal on xGMI**: its 1-stage/2-stage choice can flip between runs, so a single pass can score
  the wrong baseline. Compare across passes before trusting a small win or loss against `cdr`.
* **The fused shipped-ladder engine** (`fused_fly_1stage`) has shown slow or bimodal timings for the same
  kernel symbol that the pinned rows run fast (see `reference/xgmi_fused_results.md`). The fused fit therefore
  uses the pinned rows only.
* Run-to-run noise: very small payloads (<20 us) are sensitive to host load. The fit uses `median us`
  (median across ranks), not the max.
* Adding a kernel knob or candidate: add it to `CANDIDATES` in `../bench_comm_ar.py`, then to the
  candidate lists at the top of `tune.sh` (`MESH_PINNED`, `ONESHOT_PINNED`, ...). The fused sweep enumerates
  `fused_fly1s` candidates automatically.
* Everything here is read-only with respect to the source tree: nothing but you edits the policy tables.

## Layout

```
tune.sh                     driver (setup/check/sweep/fit/fused-*/validate/audit/report)
fit_allreduce_policy.py     plain fit + --audit-auto (imports CANDIDATES from ../bench_comm_ar.py)
shapes/                     plain sweep shape sets (M,K,label)
fused/analyze_fused.py      per-shape summary of the fused sweep
fused/fit_fused.py          fused ladder / window fit
fused/shapes/               fused sweep shape sets, one per hidden size
reference/                  results of the last xGMI (MI325X, gfx942) tuning, for comparison with a new run
out/                        sweep data, fits, logs (git-ignored; large)
```

`reference/` holds the 2026-09 MI325X/xGMI outputs: `xgmi_plain_fast_fit.txt` (plain fit),
`xgmi_plain_audit.txt` (audit), `xgmi_plain_summary.md` (speedup summary), `xgmi_fused_fit.txt` and
`xgmi_fused_results.md` (fused). A fresh run on comparable hardware should reproduce them up to noise.
