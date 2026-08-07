# Triton kernel configs — rules for automated edits

Scope: **GEMM and MOE configs only.** Read this before adding, moving,
renaming, or tuning a GEMM or MOE JSON file under
`aiter/ops/triton/configs/`, or before touching
`utils/gemm_config_utils.py` or `utils/moe_config_utils.py`.

Out of scope, do not touch without an explicit request: `configs/conv/`,
`configs/hstu_attn/`, and the flat attention / GMM / MHC / MLA files at the top
of `configs/`. They have their own loaders and are unaffected by anything here.

The tree is **mid-migration** from a flat, arch-prefixed layout to a nested
`<arch>/<backend>/<op>/<d_type>/` layout. Both layouts are live, but **the legacy flat
layout is deprecated and will be removed** — treat it as read-only history, not
as a place to add things.

Two non-negotiables:

1. **Tuning values live in JSON, never in Python.** No `setdefault`, no inline
   dict literals, no arch-conditional constants, no hardcoded fallback configs.
   If a value is missing, fix the JSON.
2. **New configs go in the target layout** unless their family is still in the
   legacy directory (see §6).

`GEMM-AFP4WFP4` (gfx950 triton, gfx950/gfx1250 gluon) is the only migrated
family and the worked reference — copy its shape when in doubt.

---

## 1. Layouts

### Target layout (use for all new configs)

```
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

| Segment     | Values                                                    |
| ----------- | --------------------------------------------------------- |
| `<arch>`    | `gfx942`, `gfx950`, `gfx1250`, `gfx1151`, `gfx1200`, `gfx1201` |
| `<backend>` | `triton` or `gluon`                                        |
| `<op>`      | `gemm` or `moe`                                            |
| `<d_type>`  | `config_name.lower().replace("-", "_")` — `GEMM-AFP4WFP4` → `gemm_afp4wfp4`. The transform lives in `gemm_config_utils._dtype_dir()` |
| filename    | **no arch prefix** — the arch is the directory. The default is literally `DEFAULT.json`; specialized files keep the `<CONFIG_NAME>-` stem |

```
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
configs/gfx950/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx1250/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
```

The `<arch>/<backend>/moe/` directories exist but are empty, held open with
`.gitkeep`. Keep them. **No MOE config has been migrated and no MOE resolver
understands the nested layout yet** — see §5.

### Legacy layout — deprecated, pending removal

```
configs/gemm/<arch>-<CONFIG_NAME>[-<suffix>].json
configs/gemm/gluon/<arch>-<CONFIG_NAME>[-<suffix>].json
configs/moe/<arch>-MOE-<dtype_str>.json
configs/moe/<arch>-A8W4.json
configs/moe/<arch>-MOE_ROUTING_SIGMOID_TOPK1.json
```

Regenerate rather than trusting this listing:
`git ls-tree -r --name-only HEAD aiter/ops/triton/configs/`

Still authoritative for every family not yet migrated. For GEMM it is reached
through the fallback chain in §2; for MOE it is the *only* path that works.
The GEMM fallback is temporary — anything left in `configs/gemm/` when the
legacy candidates are dropped from `gemm_config_utils.py` will stop resolving.

---

## 2. GEMM resolution order — `get_gemm_config()`

`utils/gemm_config_utils.py` picks a directory by probing for the *default*
config file (`DEFAULT.json` in the nested layout, `<arch>-<CONFIG_NAME>.json`
in legacy) in order and taking the first hit. Specialized files are then read
from that same directory.

**`backend=None`** (what every caller uses today):

1. `configs/<arch>/triton/gemm/<d_type>/DEFAULT.json`
2. `configs/<arch>/gluon/gemm/<d_type>/DEFAULT.json`
3. `configs/gemm/<arch>-<CONFIG_NAME>.json`  *(legacy)*

**`backend="triton"|"gluon"`**:

1. `configs/<arch>/<backend>/gemm/<d_type>/DEFAULT.json`
2. `configs/gemm/<backend>/<arch>-<CONFIG_NAME>.json`  *(legacy)*
3. `configs/gemm/<arch>-<CONFIG_NAME>.json`  *(legacy)*

If nothing matches, the last legacy candidate is used and the missing-default
assertion fires there — so error messages still point at `configs/gemm/`.

The legacy candidates are marked `# TODO(satya): legacy, remove` and are
scheduled for deletion. Do not write new code that depends on them resolving.

Consequences to keep in mind:

- **A directory is chosen as a unit.** The unit is the family's `<d_type>/`
  directory. Splitting a config family across `<arch>/triton/gemm/<d_type>/`
  and legacy `configs/gemm/` silently drops the specialized files in whichever
  directory loses the probe. Move a family wholesale or not at all. Worse: a
  `<d_type>/` directory with specialized files but **no `DEFAULT.json` is
  invisible** — the probe keys only on `DEFAULT.json` and falls through to
  legacy, ignoring everything in the directory.
- **`backend=None` prefers `triton` over `gluon`.** On an arch with only a
  gluon default (currently gfx1250 `GEMM-AFP4WFP4`), lookup falls through to
  gluon. Adding `configs/gfx1250/triton/gemm/gemm_afp4wfp4/DEFAULT.json` later
  would change which file gfx1250 resolves to — verify that is intended.
- Results are cached twice: `functools.lru_cache` on the full argument
  tuple, plus a per-path cache of parsed JSON
  (`utils/core.py::load_config_json`) that also caches negative results
  (missing files). Adding a config file at runtime therefore has no effect;
  restart the process (tooling may call `load_config_json.cache_clear()`
  instead).
- `.github/scripts/select_triton_tests.py` only globs the legacy flat
  `configs/gemm/` — configs in the nested layout are invisible to CI test
  selection until it learns the `<d_type>/` layout.

Direct-path loaders bypass the resolver's directory probe. Grep for
`f"{AITER_TRITON_CONFIGS_PATH}/..."` before moving anything —
`gluon/gemm_a8w8_blockscale.py` still builds legacy `gemm/gluon/` paths by
hand (via `load_config_json`) and must be edited when its configs move.
`gluon/gemm_a8w8.py` and `gluon/gemm_afp4wfp4.py` go through
`get_gemm_config(backend="gluon")` and need no changes.

---

## 3. GEMM config file contents

Required top-level shape:

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- `M_LEQ_x` is searched ascending over `STANDARD_M_BOUNDS =
  (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)`, then
  `M_GEQ_x` descending, then `any`. A caller may override with
  `bounds=(...)`, which must be strictly increasing positive ints.
- `any` must exist unless every reachable `M` is covered by an explicit bound.
- The deprecated `{"large": ..., "small": ...}` shape must not be introduced.
- A `KeyError` at lookup time means no bound matched — usually a missing `any`.

Each `M_*` entry carries at minimum:

```
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim,
cache_modifier, NUM_KSPLIT
```

`add_default_gemm_config_params()` backfills `NUM_KSPLIT=1` and
`cache_modifier=None` as a last resort, and `compute_splitk_params()` derives
`SPLITK_BLOCK_SIZE` and may clamp `BLOCK_SIZE_K` / `NUM_KSPLIT`. Neither is a
license to omit keys.

`get_gemm_config()` returns `(config, is_tuned)`. `is_tuned` is `True` only when
a specialized (N/K-, B-, or `specialized_filename`-keyed) file was hit, `False`
for the default file or `any`. Do not discard it.

### The JSON is the only place tuning values live

A `_get_config()` should do nothing but call `get_gemm_config()` and return:

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-AFP4WFP4", M, N, K)
```

`_triton_kernels/gemm/basic/gemm_afp4wfp4.py` carried a block of `setdefault`
calls and it was deleted — it masked incomplete config files with values nobody
had tuned, and made the effective config un-inspectable from the JSON.

---

## 4. GEMM naming

| Kind              | Target layout                                   | Legacy layout                                        |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------- |
| Default           | `gemm_a16w16/DEFAULT.json`                       | `gfx950-GEMM-A16W16.json`                             |
| N/K specialized   | `gemm_a16w16/GEMM-A16W16-N=256-K=7168.json`      | `gfx950-GEMM-A16W16-N=256-K=7168.json`                |
| Batched (B, N, K) | `batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json` | `gfx1250-BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json` |
| Custom suffix     | `fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json` | same, arch-prefixed         |

The `<d_type>` directory name is `config_name.lower().replace("-", "_")`.
Dashes, underscores, and case all fold together, so new config names must stay
distinct under that transform — `GEMM-FOO-BAR` and `GEMM-FOO_BAR` would collide.

Config-name patterns: `GEMM-A{x}W{y}`, `BATCHED_GEMM-A{x}W{y}`,
`FUSED-GEMM-{op}`, `FF-A{x}W{y}-fused`; variant suffixes
`_PRESHUFFLED`, `_BLOCKSCALE`.

**`K` in AFP4WFP4 filenames is the logical K, i.e. `2 * K_bytes`.** The kernel
does `K = 2 * K` before calling `get_gemm_config`. Tuning output that names
files by the packed byte width will never be found.

---

## 5. MOE configs

MOE does **not** go through `get_gemm_config()`. There is no probe order, no
nested-layout support, and no `is_tuned` signal. Three independent loaders read
`configs/moe/` directly, each with its own schema:

| Loader | File | Schema |
| ------ | ---- | ------ |
| `utils/moe_config_utils.py::get_moe_configs` | `moe/<arch>-MOE-<dtype_str>.json` | `small_M` / `medium_M` / `large_M` |
| `moe/moe_op_gemm_a8w4.py::_get_a8w4_dispatch` | `moe/<arch>-A8W4.json` | `bm<block_m>_n<N>_k<K>` |
| `_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py` | `moe/<arch>-MOE_ROUTING_SIGMOID_TOPK1.json` | `N16` → `small` / `medium` / … |

`<dtype_str>` comes from `get_config_dtype_str()`: `DEFAULT`, `FP8_W8A8`,
`INT8_W8A16`, `INT8_W8A8`, `INT4_W4A16`, `MX_FP4`.

`small_M` / `medium_M` / `large_M` split on `M_THRESHOLD_SMALL = 256` and
`M_THRESHOLD_MEDIUM = 1024`, both module constants in `moe_config_utils.py`.
This is **not** the GEMM `M_LEQ_x` / `M_GEQ_y` scheme — do not mix them.

### MOE is the main offender for tuning values in Python

Fix these as you touch them; do not add more:

- `get_optimal_moe_config()` returns a hardcoded dict (`BLOCK_SIZE_M: 256`,
  `BLOCK_SIZE_N: 256`, …) when no config file exists, behind a
  `warnings.warn`. A missing config silently runs untuned values.
- `moe_op_gemm_a8w4.py` has a three-tier Python fallback: exact
  `bm_n_k` hit → any-`block_m` proxy with matching `(N, K)` → a gfx942-gated
  shape heuristic → a conservative default. Only the first tier reads tuned
  numbers from JSON.

### Planned: `get_moe_config()` — design, not yet implemented

**Status: design only. Nothing below exists in the tree yet.** Do not assume
`get_moe_config` is importable; do not move MOE JSON files in anticipation of
it. If you are asked to implement it, follow this shape.

The unification is **path resolution only.** The three MOE schemas stay as they
are — the loader finds and parses the file, each caller keeps interpreting its
own structure. Converging the schemas is a separate, later decision (it would
touch every MOE config file and require re-validating dispatch on every arch).

**Step 1 — extract the shared probe.** The candidate-directory logic currently
inlined in `_get_gemm_config_cached()` becomes a helper in `utils/`, parameterised
on `<op>`:

```python
def resolve_config_dir(op: str, config_name: str, backend: str | None = None,
                       legacy_dir: str | None = None) -> tuple[str, str]:
    """Return (cfg_dir, name_prefix) for the first candidate whose default
    file exists: DEFAULT.json when name_prefix is empty (the nested layout,
    dir from _dtype_dir()), else <name_prefix><config_name>.json. Falls back
    to the last candidate so the missing-file assertion names a legacy path."""
```

Candidates for `op="moe"`, mirroring §2:

- `backend=None` → `<arch>/triton/moe/<d_type>/DEFAULT.json`,
  `<arch>/gluon/moe/<d_type>/DEFAULT.json`, then legacy `configs/moe/` with
  the `<arch>-` prefix
- `backend=...` → `<arch>/<backend>/moe/<d_type>/DEFAULT.json`, then legacy
  `configs/moe/` prefixed

`<d_type>` comes from the unprefixed stem via the same transform:
`MOE-FP8_W8A8` → `moe_fp8_w8a8`, `A8W4` → `a8w4`,
`MOE_ROUTING_SIGMOID_TOPK1` → `moe_routing_sigmoid_topk1`.

`gemm_config_utils.py` is then refactored onto the same helper with `op="gemm"`
and `legacy_dir="gemm"` — behaviour-identical, and the legacy candidates stay
tagged `# TODO(satya): legacy, remove` so both ops retire together.

**Step 2 — the loader.**

```python
def get_moe_config(config_name: str, backend: str | None = None) -> dict | None:
    """Load a MOE config by name. Returns the parsed JSON (a deep copy, safe to
    mutate), or None if no file exists for this arch. Schema interpretation is
    the caller's job — MOE files are not uniform."""
```

- `config_name` is the unprefixed stem: `MOE-FP8_W8A8`, `MOE-DEFAULT`, `A8W4`,
  `MOE_ROUTING_SIGMOID_TOPK1`.
- Cache with `functools.lru_cache` and deep-copy on return, exactly as
  `get_gemm_config()` does — callers mutate configs.
- Return `None` rather than raising: unlike GEMM, a missing MOE config is
  currently normal (only gfx942 and gfx950 ship any).
- No `is_tuned` flag. MOE has no default-vs-specialized distinction to report.

**Step 3 — port the three loaders** onto it, one PR each, without changing
schemas or file locations:

| Loader | Call becomes |
| ------ | ------------ |
| `moe_config_utils.py::get_moe_configs` | `get_moe_config(f"MOE-{dtype_str}")` |
| `moe_op_gemm_a8w4.py::_get_a8w4_dispatch` | `get_moe_config("A8W4") or {}` |
| `moe_routing_sigmoid_top1_fused.py` | `get_moe_config("MOE_ROUTING_SIGMOID_TOPK1")` |

**Step 4 — delete the hardcoded Python fallbacks.** Blocked on shipping a
`<arch>-MOE-DEFAULT.json` for the arches that have none — today only gfx942 and
gfx950 have MOE configs at all, so gfx1250/gfx1151/gfx1200/gfx1201 hit the
hardcoded dict in `get_optimal_moe_config()`. Once every supported arch has a
file, that dict and the `warnings.warn` come out.

**Step 5 — only now `git mv`** MOE files into `<arch>/<backend>/moe/` and drop
the arch prefix, following §6.

Moving MOE JSON before step 2 lands would silently break resolution, and the
Python fallbacks would swallow the breakage instead of surfacing it.

---

## 6. Migration playbook (GEMM)

One family = one `<arch>` × `<backend>` × `<CONFIG_NAME>`, including every
specialized file. `GEMM-AFP4WFP4` is the worked example — diff it against its
legacy form if a step is ambiguous.

1. **Scope it.** `ls configs/gemm/<arch>-<CONFIG_NAME>*.json` — the default plus
   every specialized file. All of them move together.
2. **Find every reader.** Grep for the config name and for
   `AITER_TRITON_CONFIGS_PATH` in `aiter/ops/triton/`. Hand-built paths must be
   rewritten; `get_gemm_config()` callers need no change.
3. **Move with `git mv`** so the change reviews as a rename. Strip the
   `<arch>-` prefix; the default file becomes `DEFAULT.json`, specialized
   files keep the `<CONFIG_NAME>-` stem:
   ```
   git mv configs/gemm/gfx950-GEMM-FOO.json \
          configs/gfx950/triton/gemm/gemm_foo/DEFAULT.json
   git mv configs/gemm/gfx950-GEMM-FOO-N=1-K=2.json \
          configs/gfx950/triton/gemm/gemm_foo/GEMM-FOO-N=1-K=2.json
   ```
   `mkdir` the `<d_type>/` directory first; it needs no `.gitkeep` (it is
   created populated). Create `<arch>/<backend>/{gemm,moe}/` with a `.gitkeep`
   if absent.
4. **Do not edit contents in the same commit.** Keep renames at 100% similarity;
   content changes go in a follow-up commit.
5. **Update docs.** `aiter/ops/triton/README.md` ("How config selection works",
   "Config file naming convention") and
   `aiter/ops/triton/utils/_triton/tunning/README.md` (the step that says
   `cp *.json .../configs/gemm/`) both still describe only the legacy layout.
6. **Pull any tuning values still hardcoded in Python into the JSON.** A
   migrated family must be fully described by its config files.
7. **Verify** on the target arch: config resolves, `is_tuned` is `True` for a
   shape that has a specialized file, and numerics are unchanged.
8. Leave the `# TODO(satya): legacy, remove` markers in `gemm_config_utils.py`
   until `configs/gemm/` is empty. Deleting the legacy fallback is the final
   step of the migration, not an intermediate one.

### Adding a *new* tuned config (no migration)

- **GEMM**, new arch/backend combination or a family already migrated →
  target layout, inside the family's `<d_type>/` directory (`DEFAULT.json`
  for the default; specialized files keep the config-name stem, no arch
  prefix).
- **GEMM**, family still in `configs/gemm/` → add to `configs/gemm/` with the
  arch prefix, and consider migrating the whole family in the same PR. Never
  create a lone nested file for a family whose default lives in legacy; the
  directory probe picks one directory and ignores the other.
- **MOE** → `configs/moe/` with the arch prefix, matching the schema of the
  loader that will read it. The nested layout is not wired up for MOE.

### Do not

- Rename or delete `.gitkeep` placeholder directories.
- Put an arch prefix on a file inside `<arch>/...`, or name a nested default
  anything other than `DEFAULT.json`.
- Put tuning values in `.py` files — no `setdefault`, no inline dicts, no
  arch-conditional constants, no hardcoded fallback configs.
- Mix the GEMM `M_LEQ_x`/`M_GEQ_y` scheme with the MOE
  `small_M`/`medium_M`/`large_M` scheme.
- Move MOE configs into `<arch>/<backend>/moe/` before a resolver exists.

### Not tuning configs

Two AOT code paths build directories under this tree at runtime that are **not
checked in and out of scope**:

- `configs/gemm/aot/<kernel>_M=…-N=…-K=…` — `gemm/fused/fused_gemm_afp4wfp4_a16w16.py`,
  `gemm/fused/fused_gemm_afp4wfp4_mul_add.py`
- `configs/paged_mqa_logits/aot/<kernel>` — `attention/pa_mqa_logits.py`

Both are guarded by `use_aot and os.path.exists(...)` and hold compiled-kernel
metadata, not tuning parameters. Do not create, migrate, or document them as
config directories.
