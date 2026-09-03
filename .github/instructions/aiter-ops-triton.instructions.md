---
applyTo: "aiter/ops/triton/**,op_tests/triton_tests/**,op_tests/op_benchmarks/triton/**"
---

# AITER Triton ops — PR review rules

When a change violates one of these rules, flag it and point the author to the
relevant rule — reviewers may not know these conventions yet.

## Reuse before adding

Always prefer reusing existing code over adding new code. Before a PR adds a
helper, kernel, or utility, the existing ones should have been checked:
`utils/` (`config_utils` and the per-family `*_config_utils` loaders,
shuffling, arch info, logging, `kernel_repr`), `_triton_kernels/common/`
(shared split-K reduce), and existing kernels and test helpers. Flag new code
that duplicates functionality already in the tree, even partially — the fix is
to extend or import the existing implementation, not to add a parallel copy.

`utils/` is layered on purpose: `config_utils.py` holds the shared core
(`resolve_config_dir`, `load_config_json`, the path constants) and each family
keeps its own loader module (`gemm_config_utils`, `conv_config_utils`,
`mhc_config_utils`, `moe_config_utils`, `tuned_config_utils`) on top of it.
Flag a function given a second home — a re-export, a wrapper that only
forwards to another module, or a copy of a core helper inside a family module.

## Folder structure and imports

The layout is: public wrapper modules in category folders
(`gemm/{basic,batched,feed_forward,fused}/`, `attention/`, `moe/`,
`normalization/`, `quant/`, `rope/`, `fusions/`, `comms/`, `conv/`,
`gated_delta_net/`, `kimi_delta_attn/`, `gluon/`), kernel bodies under
`_triton_kernels/` at the same relative category path, or under
`_gluon_kernels/<arch>/` at the same relative category path when the Gluon
implementation is architecture-specific. Shared machinery lives in `utils/`
and tuned JSON in `configs/`. Flag:

- New modules added flat at the top of `aiter/ops/triton/` — every new
  wrapper goes in the correct category folder.
- New launchable `@triton.jit` or `@gluon.jit` kernel bodies defined inside
  public wrapper modules — Triton bodies belong under `_triton_kernels/` and
  Gluon bodies under `_gluon_kernels/<arch>/`, mirroring the wrapper's
  category path. JIT-decorated device helpers that are called only from
  another kernel may remain with the entry kernel they support.
- Generic helpers (config loading, shuffling, arch detection, logging)
  re-implemented inside a kernel file instead of imported from `utils/`.
- New code importing via the legacy flat paths
  (`from aiter.ops.triton.gemm_a16w16 import ...`) — the categorized path is
  required; `_BACKWARD_COMPAT_MAP` exists only for old external callers.
- Relative imports (`from .foo import ...`, `from ..utils import ...`) —
  only absolute imports (`from aiter.ops.triton.<...> import ...`) are
  allowed.

## Tuned configs: JSON placement and naming

Every tuned config lives in one nested layout:
`configs/<arch>/<backend>/<op>/<d_type>/`, e.g.
`configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json`. `<op>` is `gemm`,
`moe`, `conv`, `mhc`, `attention`, `gmm` or `fusions`; `<d_type>` is
`config_name.lower().replace("-", "_")`. The flat arch-prefixed directories
and every fallback that reached them are gone. Flag:

- A config JSON added outside `configs/<arch>/<backend>/<op>/<d_type>/` — a
  re-created `configs/gemm/`, `configs/moe/` or `configs/conv/` directory, or
  a loose file at the top of `configs/`. Nothing resolves there any more.
- An arch prefix on a filename inside `configs/<arch>/...` (wrong:
  `configs/gfx950/triton/gemm/x/gfx950-GEMM-X.json`), or a default file named
  anything other than exactly `DEFAULT.json`.
- A specialized file added to a `<d_type>/` directory that contains no
  `DEFAULT.json`, for a family whose loader requires the default — the load
  raises for every shape, not just the unspecialized ones.
- A family's files split across two `<d_type>/` directories that differ only
  by the `_dtype_dir()` fold (`GEMM-FOO-BAR` and `GEMM-FOO_BAR` collide; two
  spellings of one family must not both exist).
- A config file that is both moved and content-edited in the same commit —
  moves must be pure `git mv` renames, content changes in a follow-up.
- A new `.gitkeep` under `configs/`. A `<d_type>/` directory is created
  populated; the few `.gitkeep` files left from the migration are inert
  leftovers, not placeholders to maintain.
- A new arch directory seeded from another arch without the copy being
  byte-identical and called out in the commit message. The one seeding rule in
  force is gfx950 → gfx1250, triton only — never into a gluon directory, never
  backwards into gfx950.
- `kpack` newly added to a gfx950 config. Triton's AMD backend deprecates
  `kpack` on CDNA4 — it warns and force-overrides `kpack = 1` there, and the
  parameter is slated for removal. The gfx950 tree is clean of it; gfx942 may
  still carry it, and existing RDNA (gfx1151/gfx1201/gfx1250) entries predate
  the rule, so flag additions rather than the entries already there.
- Checked-in files under `configs/gemm/aot/` or `configs/paged_mqa_logits/aot/`
  — these are runtime AOT caches, never committed.

## Tuned configs: JSON contents

Flag, inside GEMM-family config JSON:

- Top-level `"small"` / `"large"` keys — the required scheme is `M_LEQ_<x>` /
  `M_GEQ_<x>` / `"any"`.
- A new config file with no `"any"` entry (lookup raises `KeyError` for
  uncovered M unless every reachable M hits an explicit bound).
- Entries missing required params: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
  `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`,
  `matrix_instr_nonkdim`, `cache_modifier`, `NUM_KSPLIT`. (Loader backfill of
  `NUM_KSPLIT`/`cache_modifier` is a last resort, not a license to omit.)
- MOE dispatch keys (`bm<block_m>_n<N>_k<K>`) in a GEMM config or GEMM
  `M_LEQ_x`/`M_GEQ_y` keys in a MOE dispatch table — the schemes must not mix.
- For `*AFP4WFP4*` specialized filenames: `K` must be the logical K
  (`2 * K_bytes`) — the wrapper doubles K before lookup, so a file named by
  the packed byte width will never resolve.

And inside MOE dispatch tables:

- A newly tuned gluon dispatch shape that does not carry all six `m2bucket`
  suffixes (`tiny`, `small`, `medium`, `medium2`, `large`, `xlarge`) — a
  missing bucket falls through to `bm<block_m>_any` and silently loses that
  shape's tuning for that M range. (Existing entries are unevenly covered;
  flag new gaps, not the ones already shipped.)
- A gluon dispatch file with no `bm<block_m>_any` tier for a `block_m` it
  otherwise covers: that tier is the last resort for an unmeasured shape.
- A `BLOCK_SIZE_M` / `block_m` key inside a dispatch entry — `block_m` is the
  dispatch key (routing decides it), not a tunable.
- Triton-shaped entry keys (`BLOCK_SIZE_N`, `num_stages`, ...) in a gluon
  dispatch file or gluon-shaped keys (`block_n`, `num_buffers`,
  `persistent_iters`) in a triton one — the two paths read disjoint params.

## Python-side config hygiene

These rules apply equally to Triton and Gluon wrappers and kernels. Tuning
values for either backend live in JSON, never in Python. Flag:

- Hardcoded tuning values in `.py` files: `config.setdefault(...)` blocks,
  inline dict literals with `BLOCK_SIZE_*`/`num_warps`/`waves_per_eu` keys,
  arch-conditional tuning constants, or hardcoded fallback configs. The fix is
  always in the JSON file, not the Python.
- A kernel-level `_get_config()` whose `backend` parameter defaults to `None`
  (or any value outside `("triton", "gluon")`) — `None` is not a backend and
  `resolve_config_dir()` asserts on it, so the kernel raises the moment a
  caller omits the argument. Kernel-level helpers default to `"triton"`;
  public wrappers that expose `backend: str | None = None` must normalize it
  before calling down.
- A new or modified Triton or Gluon GEMM `_get_config()` that does anything
  beyond calling `get_gemm_config(...)` with the appropriate backend selection
  (plus `compute_splitk_params()` for split-K kernels), and that does not
  preserve the standardized `(config, is_tuned)` result:

  ```python
  # Correct — thin wrapper
  def _get_config(M: int, N: int, K: int):
      return get_gemm_config("GEMM-A16W16", M, N, K)
  ```

- A `_get_config()` that swallows the `is_tuned` flag: the standardized
  signature returns `(config, is_tuned)` straight from `get_gemm_config()`,
  so flag new or modified `_get_config()` implementations that return a bare
  config dict. Call sites may legitimately ignore the flag
  (`config, _ = _get_config(...)` is fine) — it exists so callers and tuning
  tooling can detect a shape resolving to the untuned default and log or
  re-tune.
- Raw config-file reads — `json.load(open(...))` or function-attribute caches
  like `_get_config._config_dict` — instead of
  `aiter.ops.triton.utils.config_utils.load_config_json` (which caches per
  path, including negative results) or a family loader. All hand-rolled
  loaders were deliberately removed; do not add them back.
- Mutating the dict returned by `load_config_json()` — it is the shared cached
  object. Copy first (`dict(...)` for flat entries, `copy.deepcopy` for nested
  ones); the family loaders already copy on the caller's behalf.
- New hand-built config paths (`f"{AITER_TRITON_CONFIGS_PATH}/..."`) where a
  family loader or `resolve_config_dir()` would work — a hand-built path is a
  second place the layout is encoded, and it skips the argument validation
  that makes a wrong value fail closed.
- A second MOE config reader. `utils/moe_config_utils.py::get_moe_dispatch` is
  the only MOE fetcher; flag any new MOE path built by hand, any direct
  `load_config_json` on a `moe/` file, and any reintroduced per-wrapper MOE
  loader.
- A new arch- or backend-fallback chain inside a loader (try this arch, then
  that one; try triton, then gluon). Resolution is deterministic. MHC's gfx942
  fallback is the one documented exception and it goes through the `arch=`
  override, not through a probe.

## Weight & scale shuffling — must come from `utils/shuffle.py`

All weight/scale pre-shuffle helpers are unified in
`aiter/ops/triton/utils/shuffle.py`: `shuffle_weight`,
`moe_weight_decode_view`, `shuffle_scale_gemm`, `unshuffle_scale_gemm`,
`shuffle_scale_moe`, `shuffle_scale_batched`. Flag:

- Any new local re-implementation of a weight or scale shuffle in a kernel
  wrapper, kernel body, test, or benchmark — the telltale is a small helper
  doing `view(...) → permute(...) → contiguous()` on weights or scales.
  Per-kernel copies were deliberately deleted; new layouts belong in
  `utils/shuffle.py`.
- Use of the removed name `moe_weight_gfx1250_decode_view` — it was renamed
  to `moe_weight_decode_view`.
- Hardcoded `SWIZZLE_MX_SCALE` labels (`"CDNA4_SCALE"`, `"GFX1250_SCALE"`)
  next to a `shuffle_scale_moe` call — use
  `shuffle_scale_moe(..., return_layout=True)` so the caller stays
  arch-agnostic.

## Kernel conventions

- Every new launchable Triton or Gluon kernel must set a config-aware `repr`
  using `make_kernel_repr` from `aiter.ops.triton.utils._triton.kernel_repr`:

  ```python
  _kernel_repr = make_kernel_repr(
      "_kernel_name",
      ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "num_warps"],
  )

  @triton.jit(repr=_kernel_repr)  # Triton entry kernel
  # or
  @gluon.jit(repr=_kernel_repr)   # Gluon entry kernel
  ```

  Flag every new launchable `@triton.jit` or `@gluon.jit` kernel without
  `repr=`. Include all meaningful compile-time/tuned config keys so trace
  names identify the specialization. JIT-decorated device helpers that cannot
  be launched independently do not require their own `repr`.
- Split-K GEMM implementations must use the shared second-stage reduce,
  regardless of whether the first-stage kernel uses Triton or Gluon —
  `_gemm_splitk_reduce_kernel` / `_batched_gemm_splitk_reduce_kernel` from
  `aiter/ops/triton/_triton_kernels/common/splitk_reduce.py`. Flag any new
  Triton or Gluon per-operation reduce kernel that duplicates it.
- Triton and Gluon kernel bodies are internal: they must be launched only
  from their public wrapper under `aiter/ops/triton/`. Flag any direct
  import or launch of a kernel from `_triton_kernels/` or `_gluon_kernels/`
  in tests, benchmarks, or code outside the wrapper layer — and flag new
  kernels that ship without a public wrapper.
- Arch handling: flag product names (`MI300`, `MI350`, `MI355`, ...) in
  identifiers, filenames, comments-as-logic, or any parsing of product
  strings. Compare architecture identifiers instead:

  ```python
  # Correct
  if DEVICE_ARCH in ("gfx950", "gfx1250"): ...
  # Wrong
  if int(DEVICE_ARCH.split("MI")[1]) >= 350: ...
  ```

- Flag new public wrapper functions without a docstring covering: what the
  kernel computes, the arguments (including which config parameters apply),
  the return value, and special considerations (layout expectations,
  unsupported options).

## Tests and benchmarks

- Every new Triton or Gluon kernel must ship a unit test under
  `op_tests/triton_tests/<category>/`, mirroring the kernel's category (a new
  `gemm/basic/` kernel gets `op_tests/triton_tests/gemm/basic/test_<op>.py`).
  Flag PRs that add a kernel wrapper without adding or extending a matching
  test. Likewise flag a new kernel with no benchmark script — a kernel ships
  as kernel + wrapper + unit test + benchmark.
- Tests must follow the existing pattern: pytest-style `test_<op>.py` files
  inside the category folder, shared helpers in the existing
  `*_test_utils.py` / `utils/` modules. Flag tests added flat at the
  `op_tests/triton_tests/` root or as one-off scripts.
- No kernel tuning configs in test files: flag test code that hardcodes
  config dicts (`BLOCK_SIZE_*`, `num_warps`, `waves_per_eu`, ...) or passes
  literal `config=` overrides to a wrapper. Tests exercise the wrapper's own
  config resolution — tuning values live only in `configs/` JSON.
- Benchmarks live in `op_tests/op_benchmarks/triton/` as `bench_<op>.py`,
  structured like the existing files. The config and shuffle rules above
  apply to them too: no hardcoded tuning dicts, shuffles imported from
  `aiter.ops.triton.utils.shuffle`.

## Keeping this file and the README current

Any cleanup or refactor that changes a convention under `aiter/ops/triton/` —
renaming or moving a shared helper, changing the config layout or loader
behavior, reorganizing folders, unifying duplicated code — must update
`aiter/ops/triton/README.md` **and** this instructions file in the same PR.
Flag convention-changing PRs that leave either document stale.
