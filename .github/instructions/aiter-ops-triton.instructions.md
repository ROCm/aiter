---
applyTo: "aiter/ops/triton/**,op_tests/triton_tests/**,op_tests/op_benchmarks/triton/**"
---

# AITER Triton ops — PR review rules

When a change violates one of these rules, flag it and point the author to the
relevant rule — reviewers may not know these conventions yet.

## Reuse before adding

Always prefer reusing existing code over adding new code. Before a PR adds a
helper, kernel, or utility, the existing ones should have been checked:
`utils/` (config loading, shuffling, arch info, logging, `kernel_repr`),
`_triton_kernels/common/` (shared split-K reduce), and existing kernels and
test helpers. Flag new code that duplicates functionality already in the
tree, even partially — the fix is to extend or import the existing
implementation, not to add a parallel copy.

## Folder structure and imports

The layout is: public wrapper modules in category folders
(`gemm/{basic,batched,feed_forward,fused}/`, `attention/`, `moe/`,
`normalization/`, `quant/`, `rope/`, `fusions/`, `comms/`, `conv/`,
`gated_delta_net/`, `kimi_delta_attn/`, `gluon/`), kernel bodies under
`_triton_kernels/` (or `_gluon_kernels/`) at the same relative path, shared
machinery in `utils/`, tuned JSON in `configs/`. Flag:

- New modules added flat at the top of `aiter/ops/triton/` — every new
  wrapper goes in the correct category folder.
- New `@triton.jit` kernel bodies defined inside wrapper modules — the body
  belongs under `_triton_kernels/` (Gluon: `_gluon_kernels/`) mirroring the
  wrapper's category path.
- Generic helpers (config loading, shuffling, arch detection, logging)
  re-implemented inside a kernel file instead of imported from `utils/`.
- New code importing via the legacy flat paths
  (`from aiter.ops.triton.gemm_a16w16 import ...`) — the categorized path is
  required; `_BACKWARD_COMPAT_MAP` exists only for old external callers.
- Relative imports (`from .foo import ...`, `from ..utils import ...`) —
  only absolute imports (`from aiter.ops.triton.<...> import ...`) are
  allowed.

## Tuned configs: JSON placement and naming

The config tree is mid-migration from a legacy flat layout
(`configs/gemm/<arch>-<NAME>.json`) to a nested layout
(`configs/<arch>/<backend>/<op>/<d_type>/`, e.g.
`configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json`). The legacy layout is
deprecated. Flag:

- A new GEMM config JSON added under legacy `configs/gemm/` when the family
  already has a nested `<arch>/<backend>/gemm/<d_type>/` directory, or a brand
  new family added to the legacy layout instead of the nested one.
- A config family split across the two layouts (e.g. `DEFAULT.json` nested but
  new specialized files in `configs/gemm/`, or vice versa). The directory
  probe in `get_gemm_config()` picks ONE directory — files in the losing
  directory are silently ignored. Families move wholesale or not at all.
- An arch prefix on a filename inside `configs/<arch>/...` (wrong:
  `configs/gfx950/triton/gemm/x/gfx950-GEMM-X.json`), or a nested default file
  named anything other than exactly `DEFAULT.json`.
- A specialized file added to a nested `<d_type>/` directory that contains no
  `DEFAULT.json` — the resolver probes only for the default, so the whole
  directory is invisible.
- Any MOE config created or moved under `<arch>/<backend>/moe/`. No MOE
  resolver for the nested layout exists yet; MOE configs stay in
  `configs/moe/` with the arch prefix (see `configs/CLAUDE.md` §5).
- Deleting or renaming `.gitkeep` placeholder directories under `configs/`.
- A config file that is both moved and content-edited in the same commit —
  migrations must be pure `git mv` renames, content changes in a follow-up.
- `kpack` in a config file for gfx950 or a newer arch. Triton's AMD backend
  deprecates `kpack` starting from gfx950 — it warns and force-overrides
  `kpack = 1` there, and the parameter is slated for removal. Only gfx942
  configs may still carry `kpack`.
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
- MOE-scheme keys (`small_M`/`medium_M`/`large_M`) in a GEMM config or GEMM
  `M_LEQ_x`/`M_GEQ_y` keys in a MOE config — the schemes must not mix.
- For `*AFP4WFP4*` specialized filenames: `K` must be the logical K
  (`2 * K_bytes`) — the wrapper doubles K before lookup, so a file named by
  the packed byte width will never resolve.

## Python-side config hygiene

Tuning values live in JSON, never in Python. Flag:

- Hardcoded tuning values in `.py` files: `config.setdefault(...)` blocks,
  inline dict literals with `BLOCK_SIZE_*`/`num_warps`/`waves_per_eu` keys,
  arch-conditional tuning constants, or hardcoded fallback configs. The fix is
  always in the JSON file, not the Python.
- A `_get_config()` that does anything beyond calling `get_gemm_config(...)`
  (plus `compute_splitk_params()` for split-K kernels):

  ```python
  # Correct — thin wrapper
  def _get_config(M: int, N: int, K: int):
      return get_gemm_config("GEMM-A16W16", M, N, K)
  ```

- A `_get_config()` that swallows the `is_tuned` flag: the standardized
  signature returns `(config, is_tuned)` straight from `get_gemm_config()`,
  so flag `_get_config()` implementations that return a bare config dict.
  Call sites may legitimately ignore the flag (`config, _ = _get_config(...)`
  is fine) — it exists so callers and tuning tooling can detect a shape
  resolving to the untuned default and log or re-tune.
- Raw config-file reads — `json.load(open(...))` or function-attribute caches
  like `_get_config._config_dict` — instead of
  `aiter.ops.triton.utils.core.load_config_json` (which caches per path,
  including negative results) or the resolvers `get_gemm_config` /
  `get_tuned_kernel_config`. All hand-rolled loaders were deliberately
  removed; do not add them back.
- New hand-built config paths (`f"{AITER_TRITON_CONFIGS_PATH}/..."`) where
  `get_gemm_config()` / `get_tuned_kernel_config()` would work — hand-built
  paths break silently when the family migrates to the nested layout.

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

- Every new `@triton.jit` kernel must set a config-aware repr:
  `@triton.jit(repr=make_kernel_repr("_kernel_name", [<config keys>]))` from
  `aiter.ops.triton.utils._triton.kernel_repr`. Flag new kernels without it —
  trace names must embed the tuned config.
- Split-K GEMM kernels must use the shared second-stage reduce —
  `_gemm_splitk_reduce_kernel` / `_batched_gemm_splitk_reduce_kernel` from
  `aiter/ops/triton/_triton_kernels/common/splitk_reduce.py`. Flag any new
  per-kernel reduce kernel that duplicates it.
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

- Every new kernel must ship a unit test under
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
