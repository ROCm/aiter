# Tuning scripts: rules for automated edits

Scope: every file in `aiter/ops/triton/utils/_triton/tuning/`. The config tree these scripts write
into has its own rulebook, `aiter/ops/triton/configs/AGENTS.md`; read both before changing anything
here. Update this file in the same change as any behaviour change.

## Split between tuning and benchmarks

The tuning scripts make, prune and select configs. They do not make inputs, launch kernels or
time anything. All of that is in the kernel's benchmark,
`op_tests/op_benchmarks/triton/bench_<op>.py`: `make_bench_case` gives the inputs, the launch
with a config and the derived keys, and `utils/rotating_buffer_bench.py` gives the pool and the
timer (`RotatingBufferBench.bench_times`) and the tuner's timing settings (`TUNE_SCREEN`,
`TUNE_CONFIRM`). The pool uses the `RotatingBufferBench` defaults. Never add a second timer, input
generator or launch to this directory: change the benchmark.

## File map

| File | Role | May import |
| --- | --- | --- |
| `sweep_configs.py` | driver: plan, run worker batches, final round, install | stdlib, `_utils`, `kernels` |
| `harness.py` | worker: `--plan`, or time candidates through the benchmark and write JSONL | torch/aiter/op_tests inside functions |
| `write_best_configs.py` | install step: final-round selection with `MIN_GAIN`, also a standalone CLI | aiter inside `install()` |
| `verify_configs.py` | resolve one installed shape, then time it through the benchmark | as the worker |
| `space.py` | `SEARCH_SPACE`, key discovery, shape filters, generic rules | aiter inside functions |
| `kernels.py` | one function per kernel (its benchmark via `bench_case`, `should_skip`) + `@kernel` metadata | torch/aiter/op_tests inside the functions |
| `_utils.py` | file names, records, shape args, `FINAL_PASSES`, `MIN_GAIN` | stdlib only |

## Invariants

1. Tunable keys come from the family's `DEFAULT.json`, read through `resolve_config_dir()` and
   `load_config_json()`. No hardcoded key lists, no `json.load(open(...))` on config files.
2. Keys that are not in the family's `DEFAULT.json` are not tunable: they are never swept, and the
   installed baseline is timed with those keys stripped (recorded as `dropped_keys`), so a key the
   kernel does not take is never launched. A key without a `SEARCH_SPACE` entry stops the run with
   `Unknown config key 'X' in <DEFAULT.json>: add it to SEARCH_SPACE in space.py first`. Never pin
   or skip it silently.
3. The whole `SEARCH_SPACE` is swept. No CLI overrides, no caps, no sampling. `BLOCK_SIZE_M/N`
   values below `MIN_DOT_TILE` (8) are vector tiles: only a kernel with `vector_tiles=True` gets
   them, so the other kernels keep the lists they had. With several K steps, `should_skip_generic`
   keeps `num_stages` 1 only for a vector `BLOCK_SIZE_N`. Pruning is: the
   shape filters, `should_skip_generic`, `exceeds_lds` and `exceeds_registers` (fp32 accumulator per
   thread above `MAX_ACC_PER_THREAD`: 512 on gfx1250, 256 elsewhere) in `space.py`, then the kernel's own
   `should_skip(config)`. Every rule returns True to reject; `build_space` reads as separate
   `if ...: continue` checks. When the kernel rejects every tile that fits the shape, `build_space`
   allows block sizes above the shape for that M (`oversized_tiles_allowed` in the plan). `gluon_candidates` is only for a key whose meaning differs under
   gluon (gfx950 a8w8 blockscale `num_stages` is `NUM_STAGES`, at least 2).
4. Results always install into the plain folder, `resolve_config_dir(..., cu_num=None)`, or with
   `--cu-dir` into the `<cu_num>cu/` tree of the device's CU count (`space.config_tree`), which
   must already have its own `DEFAULT.json`; the tuner never writes one. No export directory, no
   dry run.
5. The installed file is seeded with this shape's file if it exists (keeping every bucket it does
   not replace), else it starts with only `"DEFAULT_FALLBACK": true` and `DEFAULT.json`'s
   `M_BOUNDS`: never copy untuned buckets (another shape's or `DEFAULT.json`'s) into a shape file.
   The loader serves the buckets a `DEFAULT_FALLBACK` file lacks from `DEFAULT.json`. Bounds: the
   kernel's explicit `bounds`, else the seed's `M_BOUNDS`, else `STANDARD_M_BOUNDS`.
   `--all-buckets` only chooses the M list (the family's buckets up to `TOP_BUCKET_M`, 8192). Ms
   above the largest bound are not tuned. Every installed file ships an `any`, rewritten on every
   install: a copy of the file's largest-bound bucket (`M_LEQ_8192` for the standard bounds), the
   tile tuned at the largest M. A file without that bucket gets a copy of `DEFAULT.json`'s `any`
   (the one bucket copied from `DEFAULT.json`). Never a smaller-M tile: a tuned small-M tile as
   fallback breaks kernels such as MXFP4 preshuffle at M >= 32.
   `DEFAULT.json` is never written, so its `any` never changes; only `M_LEQ_*` buckets are tuned;
   `kpack` is dropped outside gfx942 and `persistent` always. After writing, clear both loader
   caches and re-resolve every swept M; `is_tuned` must be True, otherwise the previous file is
   restored. Bucket collisions are resolved before writing (largest M wins). With `--cu-dir`
   every file named here is the CU tree's own (`DEFAULT.json`, seed, `any`, `M_BOUNDS`); nothing
   is read from another tree.
6. Data contracts: a kernel function returns `(make_case, should_skip)`; `make_case(bencher)`
   is the benchmark's `make_bench_case` and returns `(launch, rot_bufs, flops, mem)`; a result
   record's fields are listed in the `harness.py` docstring and include `cu_num`, the device's CU
   count; the plan JSON carries the family's `bounds`, `cu_num` and `cu_dir`; the worker prints
   `ready` and one line per candidate for the driver's watchdog; the plan is kept per (arch,
   backend, kernel, shape) in the runs dir and the installer only lets its candidates win; the
   worker resumes a results file only if its `ready` record has the same `pool` and `timing`
   settings, and the driver only if no record was timed on another CU count;
   `write_best_configs.py --cu-dir` uses only records timed on this device's CU count (records
   without `cu_num` are skipped); `config` in a record is the raw
   candidate. Keys a wrapper derives or clamps (`SPLITK_BLOCK_SIZE`, clamped `NUM_BUFFERS`,
   `GROUP_K`, CTAS-scaled tiles) are never recorded or written. Wrappers mutate their config
   dict, so every launch gets a deep copy; the only code that mutates a config is the
   benchmark's `config_launch` (`splitk_block_size`, `compute_splitk_params`).
7. The driver never imports torch or aiter: `arch_info` initializes the GPU on import, so key
   discovery runs in the worker (`harness.py --plan`) under `HIP_VISIBLE_DEVICES`.
8. One GPU op in flight per GPU: each worker is serial and owns one GPU. A driver may use several
   GPUs (`--gpu 0 1 2 3`, distinct): candidates are dealt round-robin and timed with
   `TUNE_SCREEN`, and every shard starts with the baseline (record key `baseline@<gpu>`). Never
   compare absolute times from two GPUs: the finalists are ranked by `us` divided by the baseline
   `us` of their own GPU (absolute time only when a GPU has no ok baseline, with a warning). Then
   the final round always runs, also on one GPU. It re-times the baseline, the ten best candidates
   of the current plan by that ratio and every candidate within 3% of the best ratio on the first
   GPU, with `TUNE_CONFIRM` (`harness.py --confirm`). It runs `FINAL_PASSES` passes, one file each
   (`<results>.final<p>.jsonl`), with the same crash/hang recovery as the sweep. The installer
   uses only these files. A finalist can win only if it is `ok` in every pass, and it replaces
   the baseline only if it is `MIN_GAIN` faster in every pass. An M without all pass files is
   skipped. Never two drivers on one GPU.
9. Call the public wrappers (`aiter.ops.triton.gemm...`) only, never `_triton_kernels` or
   `_gluon_kernels`; pass `backend=` where the wrapper takes it and refuse a backend the wrapper
   cannot select (`KernelSpec.supports`).
10. Keep it readable: short functions, descriptive names (`config`, `candidate_values`, `records`,
    `block_k`, `split_k`), minimal comments that say why, no helper that is used once.

## Adding a kernel

One function in `kernels.py`, decorated with `@kernel(config_name, ...)`: `config_name` exactly as
the kernel's `_get_config` passes it, `dims` (`("B", "M", "N", "K")` for batched), `bits` (A and B
element widths as they sit in LDS, scales not counted), `cta_split` when a key such as
`num_ctas` makes `BLOCK_SIZE_M/N` a cluster tile (returns how the kernel splits it, so the LDS
and register checks and `should_skip` judge one CTA's share), `vector_tiles=True` when a path
without `tl.dot` takes `BLOCK_SIZE_M/N` below 8 (the `gemm_a16w32` fp32 FMA kernel), `gluon_archs` / `gluon_default_archs` /
`backend_kwarg` from the wrapper, `bounds` copied from the kernel's `_get_config` (the CLI K is the
logical K, so no K transform is needed). The function returns `(make_case, should_skip)`:
`make_case = bench_case("<bench module>", <shape>, <flags>)`, and `should_skip(config)` holds what
the kernel asserts (block multiples, buffer clamps). The inputs, the launch and the derived keys
go in the benchmark's `make_bench_case`, which the benchmark CLI also uses (`bench_gemm_a16w16.py`
is the example). If the benchmark has no `make_bench_case` yet, add it there; never put inputs or
a launch in `kernels.py`. The `kernels.py` docstring has a complete example. Then run the checks
below.

## Adding a key

Add it to `SEARCH_SPACE`; add an alias if a family spells it differently. Nothing else.

## Checks after any change

    export PYTHONPATH=/path/to/aiter
    python3 -c "import _utils, kernels, space, sys; assert 'torch' not in sys.modules"
    black --check . && ruff check .
    # every in-scope DEFAULT.json key is in SEARCH_SPACE
    python3 - <<'PY'
    import glob, json, kernels, space
    fams = {s.config_name.lower().replace("-", "_") for s in kernels.KERNELS.values()}
    for f in glob.glob("../../../configs/*/*/gemm/*/DEFAULT.json"):
        if f.split("/")[-2] in fams:
            for v in json.load(open(f)).values():
                if isinstance(v, dict):
                    assert all(space.ALIASES.get(k, k) in space.SEARCH_SPACE for k in v), f
    PY
    HIP_VISIBLE_DEVICES=0 python3 harness.py gemm_afp4wfp4_preshuffle --M 64 --N 7168 --K 16384 --plan --space-out /tmp/p.json
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --M 4 64 --N 256 --K 512 --gpu 0 --runs-dir /tmp/runs
    python3 sweep_configs.py gemm_afp4wfp4_preshuffle --all-buckets --N 256 --K 512 --gpu 0 1 2 3 --runs-dir /tmp/runs2
    python3 verify_configs.py gemm_afp4wfp4_preshuffle --M 64 --N 256 --K 512 --expect-tuned
    # then delete the test file configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=256-K=512.json
    # on a partition whose CU count is not the arch's default: the same sweep and verify with
    # --cu-dir (the <cu_num>cu/ tree needs its own DEFAULT.json first), then delete that file too

## Do not

- Bring back rocprof, positional config tuples, per-key CLI flags, or the old block-size group
  exclusion heuristic.
- Add a timer, a CUDA graph, input generation or a kernel launch to this directory, or timing
  flags to the command line: they live in the benchmark (`RotatingBufferBench`, `TUNE_*`).
- Install a winner from the screen numbers, or without the `MIN_GAIN` margin in every pass.
- Put an arch prefix on an installed file, write to the current directory, write `DEFAULT.json`,
  or copy untuned buckets into a shape file.
- Write into a `<cu_num>cu/` tree without `--cu-dir`, create that tree's `DEFAULT.json`, or make
  a tree for the count the plain folder was tuned on. Without `--cu-dir` the plain folder is
  used, whatever the device's CU count.
- Add tuned values to Python; `SEARCH_SPACE` is a search space, not a tuned value.
- Run two drivers on one GPU.
