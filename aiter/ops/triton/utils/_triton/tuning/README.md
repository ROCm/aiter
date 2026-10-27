# Triton / Gluon GEMM tuning scripts

Run everything from this directory with `PYTHONPATH=<aiter root>` (the benchmarks and the input
generators live in `op_tests`). `python3 sweep_configs.py --help` lists the kernels.

These scripts only make, prune and select configs. Everything about running and timing a kernel
is in its benchmark, `op_tests/op_benchmarks/triton/bench_<op>.py`: the inputs, the launch with a
config, the keys the wrapper derives, the rotating-buffer pool, the timer and the output check.
The tuner and the benchmark CLI use the same code, so a tuned config is timed the same way that
the benchmark times it.

The flow of one sweep: load the family's `DEFAULT.json` (which keys are tunable) -> generate
all combinations from `SEARCH_SPACE` -> prune -> time every survivor with the benchmark ->
re-time the fastest in a final round -> select the winner per M -> validate and install the
config file.

| Question | Answer |
| --- | --- |
| Where do I add a kernel? | A `make_bench_case` in its benchmark, then one function in `kernels.py` decorated with `@kernel(...)`; the `kernels.py` docstring shows a complete example |
| Where do I change candidate values? | `SEARCH_SPACE` in `space.py` |
| Why was this config skipped? | `space.py`: the shape filters, `should_skip_generic` and `exceeds_lds`, plus the kernel's own `should_skip` in `kernels.py`. The driver prints how many combinations each rule removed |
| How is a candidate timed? | `bench_config` and the `TUNE_*` settings in `op_tests/op_benchmarks/triton/utils/benchmark_utils.py` |
| How did this winner get installed? | `write_best_configs.py`: `load_winners` -> `assign_buckets` -> `build_table` -> write -> confirmation through `get_gemm_config` |

| File | What it does |
| --- | --- |
| `sweep_configs.py` | Driver: builds the search space for each M, sweeps it, runs the final round, then installs the best configs into the config tree |
| `harness.py` | Worker: `--plan` builds the space; otherwise times a JSON list of candidates with the kernel's benchmark and writes JSONL |
| `write_best_configs.py` | Install step; the driver runs it, rerun it by hand to re-install from existing results |
| `verify_configs.py` | Resolves the installed config for one shape through the real loader, then times and checks it with the benchmark |
| `space.py` | `SEARCH_SPACE`: the candidate values per config key; the only place that decides what is swept |
| `kernels.py` | One function per kernel: which benchmark runs it and its own `should_skip(config)` rules, registered with `@kernel(...)` (config family, backends, M bounds, element widths) |
| `_utils.py` | Small helpers, the final-round pass count and the install margin |

## Quick start

    export PYTHONPATH=/path/to/aiter
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --M 1536 --N 7168 --K 16384 --gpu 0
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --all-buckets --N 7168 --K 16384 --gpu 0 1 2 3
    python3 verify_configs.py gemm_afp4wfp4_preshuffle --M 64 --N 7168 --K 16384

`--M` sweeps the listed M values; `--all-buckets` sweeps every M bucket of the family up to 8192
(`TOP_BUCKET_M` in `_utils.py`); `any` is never swept. Batched kernels take `--B`. `K` is the
logical K, as the input generators take it (for fp4 families that is twice the byte width). Any
shape works: the generators take any M, N and K, and the unit tests' shape lists are not used.

`--gpu` takes one or more GPUs. With several, the candidates of each M are dealt round-robin to
the GPUs, one serial worker per GPU, and every GPU times the installed baseline first. GPUs of
one node do not run at exactly the same speed, so the final round picks its candidates by their
time relative to the baseline of the GPU that timed them, not by absolute time. The final round
always runs on the first GPU (see [Timing](#timing)). Never point two drivers at the same GPU,
and never list a GPU twice.

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
exceed the arch's LDS is never compiled; each kernel declares its element widths as `bits=(a, b)`,
and a kernel whose `num_ctas` splits the tile over a CTA cluster declares `cta_split`, so the
check counts one CTA's share; and a register check: a config whose fp32 accumulator per thread,
BLOCK_M x BLOCK_N / (num_warps x wave size), exceeds `MAX_ACC_PER_THREAD` (512 on gfx1250, 256
elsewhere) only spills, so it is never run. On gfx1250 1024 per thread measured 16-19x slower)
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

The worker calls the benchmark's `make_bench_case` once per shape. It builds a pool of input sets
of at least `TUNE_ROT_BUFS["target_mb"]` (512 MB, twice the 256 MB MI355X Infinity Cache), so each
launch reads cold inputs. `bench_config` captures the launches into a CUDA graph that rotates over
the pool (`do_bench_aiter_triton_times`) and records the median, min and max per launch, and
TFLOPS. The settings are in `benchmark_utils.py`, not on the command line:

| | `TUNE_SCREEN`: every candidate | `TUNE_CONFIRM`: final round, `verify_configs.py` |
| --- | --- | --- |
| launches per graph | 50, or the pool length if larger | 1000 |
| timed replays | 20 | 10 per pass |

The screen only ranks the candidates, each by its time divided by the baseline time of its own
GPU. The final round re-times the installed baseline, the ten best by that ratio and every
candidate within 3% of the best ratio, on the first GPU, in `FINAL_PASSES` (3) passes. Each pass times all of them once, in the same order, into
`<results>.final<p>.jsonl`, so the candidates interleave in time. In the final round each
candidate also runs once against the benchmark's torch reference (`output_ok`). The installer
uses only the final round (see [Install](#install)).

The installed config is always timed first as the baseline, through the wrapper's own
`config=None` path; only when the installed entry carries keys that `DEFAULT.json` does not have
are those stripped and the rest passed explicitly (`dropped_keys` in the record). The numbers
include dispatch gaps and both split-K kernels. The split-K atomic kernels are timed without a
memset of the output per launch, as in the benchmark: the atomics take the same time for any
value of the output.

## Results, resume, failures

Results go to `runs/sweep-<arch>-<backend>-<kernel>-[B=..-]M=..-N=..-K=..jsonl` (worker output in
`<that file>.gpu<g>.log`), one JSON line per config: `config` (the raw candidate), `status` `ok`
(`us`, `us_min`, `us_max`, `tflops`), `error` (the exception; the sweep continues), `crashed` or
`hung` (the worker died or stalled on it; the driver restarts the worker on the remaining
configs). Each GPU writes its own baseline record (key `baseline@<gpu>`), which carries
`is_tuned`, and the final-round records carry
`output_ok`. The full field list is in the `harness.py` docstring. Re-running the same command
skips configs that already have a record, provided the `TUNE_*` settings are unchanged
(otherwise the worker refuses); `--fresh` discards them. The final round always runs again. The
plan of each M is kept as `runs/plan-<arch>-<backend>-<kernel>-<shape>.json`, and only its
candidates (plus the baseline) can win at install time. Ctrl-C kills the workers before the
driver exits. `--batch` is the number of configs per worker process, `--stall` the seconds a
worker may spend on one candidate (a huge tile can compile for minutes) before it is killed,
`--setup-timeout` the time allowed before a worker is ready.

## Install

When every M is done the driver writes `<CONFIG>-[B=..-]N=..-K=..json` into
`configs/<arch>/<backend>/gemm/<family>/`, the directory the loader reads. An existing file for
that shape is updated in place; a new file holds only the tuned buckets plus
`"DEFAULT_FALLBACK": true` (and `DEFAULT.json`'s `M_BOUNDS`, if any), and the loader serves every
bucket it lacks from `DEFAULT.json`. The M bounds are the kernel's explicit bounds, else the file's
`M_BOUNDS`, else the standard list.

Each swept M replaces its `M_LEQ_<smallest family bound >= M>` bucket with the winner of its final
round:

1. Only finalists that are `ok` with `output_ok` in every pass can win.
2. The fastest of them by the median over the passes wins only if it is at least `MIN_GAIN`
   (2%) faster than the installed baseline in every pass.
3. Otherwise the baseline wins, and the bucket keeps the installed config.

An M without a complete final round is skipped. When several swept Ms share a bucket the largest
M wins. Ms above the largest bound (8192) are not tuned. Every installed file ships an `any`: a
copy of `DEFAULT.json`'s `any`, so an M above every tuned bucket gets the family fallback, never a
small-M tile. `DEFAULT.json` is never written. Every swept M is then re-read through
`get_gemm_config` and must come back `is_tuned`; if not, the previous file is put back and the
command fails. Config reads are cached per process, so restart Python to pick up a new file.

## Adding a kernel or a key

Kernel:

1. Give its benchmark a `make_bench_case(kwargs_rot_bufs, <shape>, <flags>)` that returns
   `(launch, rot_bufs, flops, mem, check)`, and let the benchmark's own `bench_gemm_fn` use it.
   `bench_gemm_a16w16.py` is the example:
   - `launch(config, **bufs)` comes from `config_launch(kernel_fn, derive_keys)`. It launches the
     public wrapper with a deep copy of `config`, with the keys that the wrapper derives added.
   - `check(config)` comes from `output_check` and compares one launch with the torch
     reference of the unit test.
2. Add one function to `kernels.py`, decorated with `@kernel(<config family>, ...)`. The decorator
   gives the dims, the element widths, where the gluon path exists and the M bounds that the
   kernel's `_get_config` passes. The function returns `(make_case, should_skip)`:
   - `make_case` is `bench_case("<bench module>", <shape>, <flags>)`.
   - `should_skip(config)` returns True for configs the kernel would reject, or is `None`.

The `kernels.py` docstring has a complete example. Key: add it to `SEARCH_SPACE` in `space.py`.
