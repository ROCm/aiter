# Tuning scripts: rules for automated edits

Scope: every file in `aiter/ops/triton/utils/_triton/tuning/`. The config tree these scripts write
into has its own rulebook, `aiter/ops/triton/configs/CLAUDE.md`; read both before changing anything
here. Update this file in the same change as any behaviour change.

## File map

| File | Role | May import |
| --- | --- | --- |
| `sweep_configs.py` | driver: plan, run worker batches, install | stdlib, `_utils`, `kernels` |
| `harness.py` | worker: `--plan` or time candidates, write JSONL | torch/aiter inside functions |
| `write_best_configs.py` | install step, also a standalone CLI | aiter inside `install()` |
| `verify_configs.py` | resolve + time one installed shape | as the worker |
| `space.py` | `SEARCH_SPACE`, key discovery, shape filters, generic rules | aiter inside functions |
| `kernels.py` | one function per kernel (inputs, launch, resets, derived keys, `should_skip`) + `@kernel` metadata | torch/aiter/op_tests inside the functions |
| `_utils.py` | file names, records, shape args | stdlib only |

## Invariants

1. Tunable keys come from the family's `DEFAULT.json`, read through `resolve_config_dir()` and
   `load_config_json()`. No hardcoded key lists, no `json.load(open(...))` on config files.
2. Keys that are not in the family's `DEFAULT.json` are not tunable: they are never swept, and the
   installed baseline is timed with those keys stripped (recorded as `dropped_keys`), so a key the
   kernel does not take is never launched. A key without a `SEARCH_SPACE` entry stops the run with
   `Unknown config key 'X' in <DEFAULT.json>: add it to SEARCH_SPACE in space.py first`. Never pin
   or skip it silently.
3. The whole `SEARCH_SPACE` is swept. No CLI overrides, no caps, no sampling. Pruning is: the
   shape filters, `should_skip_generic` and `exceeds_lds` in `space.py`, then the kernel's own
   `should_skip(config)`. Every rule returns True to reject; `build_space` reads as separate
   `if ...: continue` checks. When the kernel rejects every tile that fits the shape, `build_space`
   allows block sizes above the shape for that M (`oversized_tiles_allowed` in the plan). `gluon_candidates` is only for a key whose meaning differs under
   gluon (gfx1250 a8w8 blockscale `num_stages` is `NUM_BUFFERS`).
4. Results always install into `resolve_config_dir(op, config_name, backend)`. No export directory,
   no dry run.
5. The installed file is seeded with what the loader serves today, in the loader's lookup order
   (this shape's file, for batched shapes the N/K file, else `DEFAULT.json`), and keeps every
   bucket it does not replace. Bounds: the kernel's explicit `bounds`, else the seed file's
   `M_BOUNDS`, else `STANDARD_M_BOUNDS`, because the loader reads `M_BOUNDS` from the file it picks.
   `--all-buckets` only chooses the M list (the family's buckets up to `TOP_BUCKET_M`, 4096); it
   never touches `any` either. `any` is never modified; `M_GEQ_*` is never generated;
   `kpack` is dropped outside gfx942 and `persistent` always. After writing, clear both loader
   caches and re-resolve every swept M; `is_tuned` must be True, otherwise the previous file is
   restored. Bucket collisions are resolved before writing (largest M wins).
6. Data contracts: a kernel function returns `(call, inputs, should_skip)`; a result
   record's fields are listed in the `harness.py` docstring; the plan JSON carries the family's
   `bounds`; the worker prints `ready` and one line per candidate for the driver's watchdog;
   the plan is kept per (arch, backend, kernel, shape) in the runs dir and the installer only
   lets its candidates win; resume requires the same `--calls`/`--replays`; `config` in a record is the raw
   candidate. Keys a wrapper derives or clamps (`SPLITK_BLOCK_SIZE`, clamped `NUM_BUFFERS`,
   `GROUP_K`, CTAS-scaled tiles) are never recorded or written. Wrappers mutate their config
   dict, so every launch gets a deep copy; the only framework code that mutates a config is a
   kernel's `call` (`add_splitk_block_size`, `compute_splitk_params`).
7. The driver never imports torch or aiter: `arch_info` initializes the GPU on import, so key
   discovery runs in the worker (`harness.py --plan`) under `HIP_VISIBLE_DEVICES`.
8. One GPU op in flight per GPU: each worker is serial and owns one GPU. A driver may use several
   GPUs (`--gpu 0 1 2 3`): candidates are dealt round-robin, then a final round re-times the
   baseline and the ten fastest on the first GPU into `<results>.final.jsonl`; when that file
   exists the installer uses only it, and an M timed on several GPUs without it is skipped. Never two
   drivers on one GPU.
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
check and `should_skip` judge one CTA's share), `gluon_archs` / `gluon_default_archs` /
`backend_kwarg` from the wrapper, `bounds` copied from the kernel's `_get_config` (the CLI K is the
logical K, so no K transform is needed). The function builds the inputs once with the `op_tests`
generator, preallocates the output, and returns `(call, inputs, should_skip)`;
`should_skip(config)` holds what the kernel asserts (block multiples, buffer clamps). Output
resets and derived keys live in `call`. Keep everything about a kernel inside that one function;
the `kernels.py` docstring has a complete example. Then run the checks below.

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

## Do not

- Bring back rocprof, positional config tuples, per-key CLI flags, or the old block-size group
  exclusion heuristic.
- Put an arch prefix on an installed file, write to the current directory, or touch `any`.
- Add tuned values to Python; `SEARCH_SPACE` is a search space, not a tuned value.
- Run two drivers on one GPU.
