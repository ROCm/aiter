# Triton / Gluon GEMM tuning scripts

Run everything from this directory with `PYTHONPATH=<aiter root>` (the input generators live in
`op_tests`). `python3 sweep_configs.py --help` lists the kernels.

The flow of one sweep: load the family's `DEFAULT.json` (which keys are tunable) -> generate
all combinations from `SEARCH_SPACE` -> prune -> benchmark every survivor -> select the winner
per M -> validate and install the config file.

| Question | Answer |
| --- | --- |
| Where do I add a kernel? | One function in `kernels.py` decorated with `@kernel(...)`; the module docstring shows a complete example |
| Where do I change candidate values? | `SEARCH_SPACE` in `space.py` |
| Why was this config skipped? | `space.py`: the shape filters, `should_skip_generic` and `exceeds_lds`, plus the kernel's own `should_skip` in `kernels.py`. The driver prints how many combinations each rule removed |
| How did this winner get installed? | `write_best_configs.py`: `load_winners` -> `assign_buckets` -> `build_table` -> write -> confirmation through `get_gemm_config` |

| File | What it does |
| --- | --- |
| `sweep_configs.py` | Driver: builds the search space for each M, times every candidate on one GPU, then installs the best configs into the config tree |
| `harness.py` | Worker: `--plan` builds the space; otherwise times a JSON list of candidates with CUDA-graph replay and writes JSONL |
| `write_best_configs.py` | Install step; the driver runs it, rerun it by hand to re-install from existing results |
| `verify_configs.py` | Resolves the installed config for one shape through the real loader and times it |
| `space.py` | `SEARCH_SPACE`: the candidate values per config key; the only place that decides what is swept |
| `kernels.py` | One function per kernel: inputs, the launch, output resets, derived keys and its own `should_skip(config)` rules, registered with `@kernel(...)` (config family, backends, M bounds, element widths) |
| `_utils.py` | Small helpers |

## Quick start

    export PYTHONPATH=/path/to/aiter
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --M 1536 --N 7168 --K 16384 --gpu 0
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --all-buckets --N 7168 --K 16384 --gpu 0 1 2 3
    python3 verify_configs.py gemm_afp4wfp4_preshuffle --M 64 --N 7168 --K 16384

`--M` sweeps the listed M values; `--all-buckets` sweeps every M bucket of the family up to 4096
(`TOP_BUCKET_M` in `_utils.py`); `any` is left as installed either way. Batched kernels take `--B`. `K` is the logical K, as the input generators take
it (for fp4 families that is twice the byte width).

`--gpu` takes one or more GPUs. With several, the candidates of each M are dealt round-robin to
the GPUs, one serial worker per GPU; when all are done a final round re-times the installed
baseline and the ten fastest candidates on the first GPU into `<results>.final.jsonl`, and the
installer uses only that file for the M, so the winner is picked from numbers measured on one
device. Never point two drivers at the same GPU.

Suggested M lists: `GEMM-AFP4WFP4_PRESHUFFLED` 4 8 16 31 32 64 ... 8192 (31 is a bucket of its
own because `BLOCK_SIZE_M` must be 16 or less below M=32); standard families 1 4 8 16 ... 8192;
`GEMM-A16W16-gated` 64 128 256 512 2048.

## How the tunable keys are found

The keys are the union of the bucket keys in the kernel family's `DEFAULT.json`
(`configs/<arch>/<backend>/gemm/<family>/DEFAULT.json`, read through the normal config loader).
Every key must have an entry in `SEARCH_SPACE`; otherwise the run stops with
`Unknown config key 'X' in ...: add it to SEARCH_SPACE in space.py first`. A key that only a
specialized file carries is not tunable until it is in `DEFAULT.json`; the installed baseline
is timed with such keys stripped (the record lists them as `dropped_keys`).

## The search space

`SEARCH_SPACE` is always swept in full: the cartesian product of its lists for the discovered keys,
minus shape filters (block sizes above the next power of two of the dimension, `NUM_KSPLIT` values
that do not divide K), the generic rules in `space.py` (the old split-K pruning rules, applied only
when the keys exist, and an LDS check: a block-size combination whose buffers x (A tile + B tile)
exceed the arch's LDS is never compiled; each kernel declares its element widths as `bits=(a, b)`)
and the kernel's own `should_skip(config)` in `kernels.py` (what the kernel asserts, and buffer
counts its wrapper clamps so they would only repeat another candidate). Every rule returns True
to reject. If the kernel rejects every tile that fits the shape (a kernel that needs 64-row tiles
at M=16), block sizes above the shape are allowed for that M and the plan says so.
There are no command-line overrides: edit the table. A one-value list pins a key. Keys the wrapper
never reads (`ignored_keys` in the spec) and `matrix_instr_nonkdim`/`kpack` under gluon stay at
their `DEFAULT.json` value. The driver prints the candidates per key, the config count and an ETA
before it starts, with the number of combinations each rule removed; a full Triton space (10 keys)
is around 10^5 configs per M, a gluon family a few hundred to a few thousand.

## Backends

`--backend` defaults to what the wrapper picks on this arch (gluon on gfx1250 for the families that
have a gluon kernel). Wrappers without a `backend=` argument, such as `gemm_afp4wfp4_preshuffle`,
cannot be forced onto the other backend; the driver refuses.

## Timing

Each candidate is launched once eagerly (this compiles it), then captured into a CUDA graph of 24
launches that rotate over cold copies of the inputs (up to `--cold-mb`), replayed 25 times. The
median per launch is recorded with min, max and TFLOPS. The installed config is always timed
first as the baseline, through the wrapper's own `config=None` path; only when the installed entry
carries keys that `DEFAULT.json` does not have are those stripped and the rest passed explicitly
(`dropped_keys` in the record). The numbers include dispatch gaps and both split-K kernels, so
they are not comparable with the old rocprof logs.

## Results, resume, failures

Results go to `runs/sweep-<arch>-<backend>-<kernel>-[B=..-]M=..-N=..-K=..jsonl` (worker output in
`<that file>.gpu<g>.log`), one JSON line per config: `config` (the raw candidate), `status` `ok` (`us`, `us_min`, `us_max`, `tflops`), `error`
(the exception; the sweep continues), `crashed` or `hung` (the worker died or stalled on it; the
driver restarts the worker on the remaining configs). The baseline record carries `is_tuned`.
The full field list is in the `harness.py` docstring. Re-running the same command skips configs that already have a record, provided `--calls` and
`--replays` are unchanged (otherwise it refuses); `--fresh` discards them. The plan of each M is
kept as `runs/plan-<arch>-<backend>-<kernel>-<shape>.json`, and only its candidates (plus the
baseline) can win at install time. Ctrl-C kills the workers before the driver exits. `--batch` is the number of configs per worker process, `--stall` the
seconds a worker may spend on one candidate (a huge tile can compile for minutes) before it is
killed, `--setup-timeout` the time allowed before a worker is ready.

## Install

When every M is done the driver writes `<CONFIG>-[B=..-]N=..-K=..json` into
`configs/<arch>/<backend>/gemm/<family>/`, the directory the loader reads. The file is seeded with
what the loader serves today, in its own lookup order: the installed file for that shape, for
batched shapes the N/K file, else `DEFAULT.json`, so no bucket loses its tuning; the M bounds are
the kernel's explicit bounds, else the seed file's `M_BOUNDS`, else the standard list. Each swept M
replaces its `M_LEQ_<smallest family bound >= M>` bucket with the fastest `ok` record, the
installed baseline included; when several swept Ms share a bucket the largest M wins. `any` is never
modified and an M above the largest bound is not written. Every swept M is then re-read
through `get_gemm_config` and must come back `is_tuned`; if not, the previous file is put back and
the command fails.
Config reads are cached per process, so restart Python to pick up a new file.

## Adding a kernel or a key

Kernel: add one function to `kernels.py`, decorated with `@kernel(<config family>, ...)` giving the
dims, the element widths, where the gluon path exists and the M bounds the kernel's `_get_config`
passes. The function generates the inputs once and returns `(call, inputs, should_skip)`:
`call(config, *inputs)` launches the public wrapper with `config=config` (and resets outputs or adds
derived keys there), `should_skip(config)` returns True for configs the kernel would reject (or is
`None`). The `kernels.py` docstring has a complete example. Key: add it to `SEARCH_SPACE` in
`space.py`.
