# Triton kernel configs — rules for automated edits

Scope: **every tuned JSON file under `aiter/ops/triton/configs/`**, and the
loader modules that read them (`aiter/ops/triton/utils/config_utils.py` and
the per-family `*_config_utils.py` modules). Read this before adding, moving,
renaming, or tuning a config file.

There is now exactly **one layout**. The flat, arch-prefixed directories
(`configs/gemm/`, `configs/moe/`, `configs/conv/`, the loose attention / GMM /
MHC files at the top of `configs/`) are gone, and so is the fallback code that
used to reach them. Nothing resolves outside the nested tree.

Two non-negotiables:

1. **Tuning values live in JSON, never in Python.** No `setdefault`, no inline
   dict literals, no arch-conditional constants, no hardcoded fallback configs.
   If a value is missing, fix the JSON.
2. **Every config read goes through a loader** in `utils/`, which goes through
   `resolve_config_dir()` + `load_config_json()`. No `json.load(open(...))`, no
   function-attribute caches, no hand-built `f"{AITER_TRITON_CONFIGS_PATH}/…"`
   paths.

---

## 1. The layout

```
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

| Segment     | Values                                                      |
| ----------- | ----------------------------------------------------------- |
| `<arch>`    | `gfx942`, `gfx950`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx1250` |
| `<backend>` | `triton` or `gluon`                                          |
| `<op>`      | `gemm`, `moe`, `conv`, `mhc`, `attention`, `gmm`, `fusions`  |
| `<d_type>`  | `config_name.lower().replace("-", "_")` — `GEMM-AFP4WFP4` → `gemm_afp4wfp4`. The transform is `config_utils._dtype_dir()` |
| filename    | **no arch prefix** — the arch is the directory. The default is literally `DEFAULT.json`; specialized files keep the `<CONFIG_NAME>-` stem |

```
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
configs/gfx1250/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx1250/gluon/moe/a8w4/DEFAULT.json
configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json
configs/gfx1201/triton/conv/conv_3x3_nhwc/DEFAULT.json
```

Regenerate rather than trusting any listing in this file:

```
git ls-tree -r --name-only HEAD aiter/ops/triton/configs/
```

A few `.gitkeep` files survive from the migration in directories that later
got real content or never got any. They are inert: no loader looks for them,
and nothing breaks if one is deleted along with an otherwise-empty directory.
Do not add new ones — a `<d_type>/` directory is created populated.

---

## 2. Resolution — `resolve_config_dir()`

```python
resolve_config_dir(op, config_name, backend="triton", arch=None) -> str
```

in `utils/config_utils.py` is the single path builder. It **builds** the
directory; it never probes, never searches, and has no fallback chain:

```
{AITER_TRITON_CONFIGS_PATH}/{arch}/{backend}/{op}/{_dtype_dir(config_name)}
```

- `arch` defaults to `arch_info.get_arch()`. The `arch=` argument is an
  explicit override for loaders that deliberately retry under another
  architecture (MHC's gfx942 fallback, §5.3) — not a search order.
- `backend` is **declared by the caller** and defaults to `"triton"`. Gluon
  kernels and gluon dispatch paths pass `"gluon"`. There is no cross-backend
  search: the two backends take disjoint config params, so a config tuned for
  the other backend is not usable, and silently borrowing one would be a bug,
  not a convenience.
- Every argument becomes a path component, so each is validated against a
  whitelist and the function **fails closed** with an `AssertionError`:

  | Argument | Pattern | Why |
  | -------- | ------- | --- |
  | `op` | `[a-z][a-z0-9_]*` | directory name in the tree |
  | `config_name` | `[A-Za-z0-9][A-Za-z0-9_-]*` | folded into `<d_type>` |
  | `backend` | one of `("triton", "gluon")` | |
  | `arch=` override | `[a-z][a-z0-9_]*` | programmer-written literal |
  | running arch | `[A-Za-z0-9][A-Za-z0-9_.+:-]*` | driver-derived; tolerates vendor formats but must stay path-safe |

- Existence is **not** checked. Whether a given file inside the directory has
  to exist is the loader's decision, expressed through `load_config_json()`.

### `load_config_json()`

```python
load_config_json(fpath, required=True) -> dict | None
```

- `required=True` (the default) raises `FileNotFoundError` naming the exact
  nested path when the file is missing. That message is the error every
  missing required table should produce — do not wrap it in a vaguer one.
- `required=False` returns `None`, for genuinely optional tables (probes,
  arch-specific extras, per-path fallbacks). Callers must handle `None`
  explicitly.
- Cached per path with `functools.lru_cache`, **including negative results**.
  Adding a config file at runtime therefore has no effect: restart the
  process, or call `load_config_json.cache_clear()` from tooling. Exceptions
  are never cached, so a missing required file raises consistently on every
  call.
- The returned dict is the **shared cached object**. Copy before mutating — a
  shallow `.copy()` for flat bucket dicts, `copy.deepcopy` when nested
  sub-dicts get mutated. The family loaders already do this for their callers.

---

## 3. Loader modules

`utils/config_utils.py` is the shared core: the config-tree paths,
`load_config_json()`, `_dtype_dir()` and `resolve_config_dir()`. Each op
family keeps its own small module built on that core. Every function has
exactly one home; there is no facade or re-export layer.

| Module | Entry points | Reads |
| ------ | ------------ | ----- |
| `utils/config_utils.py` | `resolve_config_dir`, `load_config_json`, `AITER_TRITON_CONFIGS_PATH`, `AITER_TRITON_OPS_PATH`, `USE_LRU_CACHE` | — (core) |
| `utils/gemm_config_utils.py` | `get_gemm_config`, `add_default_gemm_config_params`, `compute_splitk_params`, `pick_gemm_num_stages`, `STANDARD_M_BOUNDS` | `<arch>/<backend>/gemm/<d_type>/` |
| `utils/conv_config_utils.py` | `get_conv_config`, `has_conv_config`, `has_exact_conv_config`, `conv_config_uses_exact_routes`, `format_shape_key`, `format_prepack_shape_key`, `CONV_STANDARD_M_BOUNDS` | `<arch>/triton/conv/<d_type>/` |
| `utils/mhc_config_utils.py` | `get_mhc_config`, `get_mhc_post_config`, `hip_post_dispatch_block` | `<arch>/triton/mhc/<d_type>/` (gfx942 fallback) |
| `utils/moe_config_utils.py` | `get_moe_dispatch` | `<arch>/<backend>/moe/<d_type>/` |
| `utils/tuned_config_utils.py` | `get_tuned_kernel_config` | `<arch>/<backend>/<op>/<d_type>/DEFAULT.json` |

Attention and GMM kernels have no family module: they call
`resolve_config_dir()` + `load_config_json()` directly from their kernel file,
which is fine for a single `DEFAULT.json` read with no selection logic.

Adding a family module is the right move only when a family grows real
selection logic (bucket walks, specialized-file discovery, fallbacks). Until
then, two lines against the core beat a module.

Two places still interpolate `AITER_TRITON_CONFIGS_PATH` by hand instead of
calling the resolver: `tuned_config_utils._get_tuned_kernel_entry()` and
`fusions/fused_clamp_act_mul.py::_get_config()` (whose gfx950 fallback is
exactly the resolver's `arch=` override). Both land on the same directory the
resolver would build, but they re-encode the layout and skip the argument
validation. Move them onto `resolve_config_dir()` when you touch them; do not
add a third.

---

## 4. GEMM — `get_gemm_config()`

```python
get_gemm_config(config_name, M, N=None, K=None, bounds=None,
                specialized_filename=None, backend="triton", B=None)
    -> (config: dict, is_tuned: bool)
```

Order of operations:

1. `<arch>/<backend>/gemm/<d_type>/DEFAULT.json` — **must exist**, else
   `AssertionError` naming the path.
2. Specialized file, first hit wins:
   `{config_name}-B={B}-N={N}-K={K}.json` (when `B` is given), then
   `{config_name}-N={N}-K={K}.json`; or `{config_name}-{specialized_filename}.json`
   when the caller passes one (fused kernels with several N dims), which
   bypasses the B/N/K candidates.
3. Inside the chosen file: `M_LEQ_x` ascending, then `M_GEQ_x` descending,
   then `"any"`. `KeyError` if nothing matches.

`is_tuned` is `True` only when a specialized file was hit — `False` for the
default file and for `"any"`. **Do not discard it**; it is how callers and
tuning tooling detect a shape running on untuned numbers. Call sites that
legitimately ignore it use `config, _ = _get_config(...)`.

The returned config is a fresh deep copy, safe to mutate.

### File contents

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- `M_LEQ_x` is searched over `STANDARD_M_BOUNDS =
  (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)`. A caller may
  override with `bounds=(...)`, which must be strictly increasing positive
  ints.
- `any` must exist unless every reachable `M` is covered by an explicit bound.
  A `KeyError` at lookup time usually means it is missing.
- The deprecated `{"large": …, "small": …}` shape must not be introduced.
- Each `M_*` entry carries at minimum:

  ```
  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
  num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim,
  cache_modifier, NUM_KSPLIT
  ```

  `add_default_gemm_config_params()` backfills `NUM_KSPLIT=1` and
  `cache_modifier=None` as a last resort, and `compute_splitk_params()`
  derives `SPLITK_BLOCK_SIZE` and may clamp `BLOCK_SIZE_K` / `NUM_KSPLIT`.
  Neither is a license to omit keys.

### `_get_config()` stays a thin wrapper

```python
def _get_config(M: int, N: int, K: int, backend: str = "triton"):
    return get_gemm_config("GEMM-A16W16", M, N, K, backend=backend)
```

A kernel-level `_get_config()` that takes a `backend` argument **defaults it
to `"triton"`**, never to `None`: `None` is not a backend, and
`resolve_config_dir()` rejects it. Public wrappers that expose
`backend: str | None = None` normalize it (typically to `"gluon"` on gfx1250,
`"triton"` elsewhere) before calling down.

`_triton_kernels/gemm/basic/gemm_afp4wfp4.py` once carried a block of
`setdefault` calls; it was deleted, because it masked incomplete config files
with values nobody had tuned and made the effective config un-inspectable from
the JSON.

### Naming

| Kind              | Path                                                             |
| ----------------- | ---------------------------------------------------------------- |
| Default           | `gemm_a16w16/DEFAULT.json`                                        |
| N/K specialized   | `gemm_a16w16/GEMM-A16W16-N=256-K=7168.json`                       |
| Batched (B, N, K) | `batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json`  |
| Custom suffix     | `fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json` |

Config-name patterns: `GEMM-A{x}W{y}`, `BATCHED_GEMM-A{x}W{y}`,
`FUSED-GEMM-{op}`, `FF-A{x}W{y}-fused`; variant suffixes `_PRESHUFFLED`,
`_BLOCKSCALE`.

Dashes, underscores and case all fold together in `<d_type>`, so new config
names must stay distinct under that transform — `GEMM-FOO-BAR` and
`GEMM-FOO_BAR` would collide on one directory.

**`K` in AFP4WFP4 filenames is the logical K, i.e. `2 * K_bytes`.** The kernel
does `K = 2 * K` before calling `get_gemm_config()`. Tuning output that names
files by the packed byte width will never be found.

---

## 5. The other families

### 5.1 MOE — `get_moe_dispatch()`

```python
get_moe_dispatch(config_name, arch, backend) -> dict
```

is the **only** MOE config fetcher. It reads
`<arch>/<backend>/moe/<d_type>/DEFAULT.json` and returns `{}` when this arch
and backend ship no tuned file, so callers fall through to their own defaults
instead of crashing. `arch` is passed in by callers that already resolved it
(it keys the cache); `resolve_config_dir()` reads the same value.

The returned dict is the shared cached object — **read-only**.

The two dispatch paths key the same family differently, which is exactly why
`backend` is a caller-declared argument rather than something the loader
guesses:

| Family | Backend | Key | Entry keys |
| ------ | ------- | --- | ---------- |
| `A8W4` | `triton` | `bm<block_m>_n<N>_k<K>` | `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim` |
| `A8W4` | `gluon` | `bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any` | `block_n`, `block_k`, `num_buffers`, `num_warps`, `persistent_iters` |
| `A4W4` | `gluon` | `bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any` | `block_n`, `block_k`, `num_buffers`, `num_warps` |

- Entries omit `BLOCK_SIZE_M` / `block_m` on purpose: `block_m` is the
  dispatch **key**, decided by routing, not a tunable.
- `<bucket>` is `m2bucket()`, splitting M on 8 / 32 / 128 / 256 / 512 into
  `tiny` / `small` / `medium` / `medium2` / `large` / `xlarge`.
  `moe_op_gemm_a8w4.py` and `moe_op_gemm_a4w4.py` currently carry identical
  copies of it — if you touch one, they belong in a shared home, not diverged.
  A missing bucket falls all the way through to `bm<block_m>_any`, which
  loses that shape's tuning entirely — so a newly tuned shape should carry all
  six, even where values repeat. Coverage in the shipped tables is uneven
  (`a4w4` is complete; most `a8w4` gluon shapes cover only the M range they
  were measured over), which is a gap to close, not a pattern to copy.
- The `bm<block_m>_any` tier must exist in every gluon dispatch file — it is
  the last resort for an unmeasured shape.
- `moe_op_gemm_a8w4.py`'s triton path additionally derives a proxy from a
  tuned entry with the same `(N, K)` under a different `block_m` before
  reaching its Python default. That proxy reads tuned numbers out of JSON; the
  final default does not, and is the tier to delete as coverage grows.

### 5.2 Conv — `get_conv_config()`

```python
get_conv_config(config_name, shape_key=None, M=None, variants=()) -> dict
```

reads `<arch>/triton/conv/<d_type>/DEFAULT.json` (conv is triton-only) and
walks four tiers, first hit wins:

1. `shapes_<variant>[shape_key]` — optional variant-specific pin
   (`shapes_nchw`, `shapes_nhwc`).
2. `shapes[shape_key]` — generic exact-shape pin.
3. `M_LEQ_<n>` — bucket walk over `CONV_STANDARD_M_BOUNDS`
   (`4 … 262144`), on `M_total` for the GEMM-like kernels, `T` for Winograd.
4. `"any"` — global fallback.

Shape keys are built by `format_shape_key()`
(`N=…,C=…,H=…,W=…,K=…,R=…,S=…,sh=…,sw=…,ph=…,pw=…,dh=…,dw=…`) and
`format_prepack_shape_key()` (`N=…,C=…,H=…,W=…,CB=…`) — never hand-formatted at
a call site. `route_exact_only: true` in a file restricts routing to exact
entries, read through `conv_config_uses_exact_routes()`.

`has_conv_config()` probes whether the running arch ships an optional table at
all (it passes `required=False`); the other entry points require the file.
Families: `CONV-1X1`, `CONV-3X3-NHWC`, `CONV-3X3-CBLOCKED`, `CONV-3X3-NCHW`,
`CONV-GENERAL`, `CONV-PREPACK`, `CONV-WINO-F4X3-{INPUT,GEMM,OUTPUT}`.

### 5.3 MHC — `get_mhc_config()`

```python
get_mhc_config(config_name, M, C, mode=None) -> (config, used_specialized)
get_mhc_post_config(M, C) -> dict
```

`mode` is required in practice: anything other than `"sinkhorn"` — including
the `None` default — raises `ValueError`.

The family directory is `<arch>/triton/mhc/<d_type>/`, with `<d_type>` derived
from `f"{config_name}_{mode.upper()}"` — `MHC_FUSED` + `sinkhorn` →
`mhc_fused_sinkhorn`. Selection is C first, then M:

- C: the largest `-C=<value>` file threshold `<= C` wins. Available thresholds
  are **discovered by globbing** `MHC_FUSED_SINKHORN-C=*.json` — in the running
  arch's directory *and* gfx942's, unioned — so a new specialized file becomes
  reachable just by being added (subject to the load cache), and a threshold
  that exists only under gfx942 is still a candidate on other arches.
- M: within the selected file, the largest `M_LEQ_<x> <= M`, else `"any"`.

**Arch fallback:** an arch with no MHC directory falls back to the `gfx942`
files, via the `arch=` override on `resolve_config_dir()`. This is a
deliberate, documented exception — it keeps MHC running (possibly
suboptimally) on untuned hardware. Do not copy the pattern into other
families without the same explicit justification.

`get_mhc_post_config()` reads `mhc_post/DEFAULT.json` and picks the largest
`C_<value> <= C`, else `"default"`. `hip_post_dispatch_block()` mirrors
`MHC_POST_KERNEL_DISPATCH` in `csrc/kernels/mhc_kernels.cu` and belongs next
to it in review.

### 5.4 Pinned autotune tiles — `get_tuned_kernel_config()`

```python
get_tuned_kernel_config(op, config_name, kernel_name, fallback, backend="triton")
    -> triton.Config
```

For kernels whose autotune search space lives in Python and only need **one
pinned tile per device**. It reads `<op>/<d_type>/DEFAULT.json` for the running
arch and looks up `kernel_name` inside it, returning `fallback` (with a
warning) when the arch publishes no entry.

The `fallback` must be **launchable anywhere**, not fastest somewhere: the same
tile can fit in 16 KB of LDS on one arch and overflow another's 64 KB. An
unmeasured device stays on the fallback until a measured entry is published.

File shape: `{"<kernel_name>": {tile keys…, "num_warps": n, "num_stages": n}}`.

---

## 6. Adding a config

1. **Name the family.** Pick the `<CONFIG_NAME>`, check `<d_type>` does not
   collide with an existing directory under that arch/backend/op.
2. **Put the default in place**: `<arch>/<backend>/<op>/<d_type>/DEFAULT.json`.
   No arch prefix in the filename. A `<d_type>/` directory whose required
   default is missing raises at first lookup naming that exact path — that is
   the intended failure, not something to paper over with a fallback.
3. **Specialized files** keep the `<CONFIG_NAME>-` stem next to the default:
   `GEMM-A16W16-N=256-K=7168.json`, `MHC_FUSED_SINKHORN-C=128.json`.
4. **Pull any tuning value still hardcoded in Python into the JSON.** A family
   must be fully described by its config files.
5. **Verify on the target arch**: the config resolves, `is_tuned` is `True`
   for a shape that has a specialized file, and numerics are unchanged.
6. If the change moves or renames files, keep the commit a pure `git mv`
   (100% rename similarity) and put content edits in a follow-up commit.

### Seeding a new architecture

An arch that ships no file for a family gets whatever that family's loader
does on a miss — a hard error for GEMM/conv/MHC required tables, `{}` for the
MOE dispatch, the `fallback` tile for `get_tuned_kernel_config()`. Where that
is not acceptable, seed the directory with a **byte-identical copy** from the
closest measured arch and say so in the commit message.

The one seeding rule currently in force: **gfx950 → gfx1250, triton only.**
Never seed a gluon directory from another arch (gluon configs carry
arch-specific tile and buffer counts), and never seed backwards into gfx950.
A seed is a placeholder that unblocks the caller-declared backend policy — it
is not tuning, and it should be replaced by measured numbers.

### Do not

- Put an arch prefix on a file inside `<arch>/…`, or name a default anything
  other than `DEFAULT.json`.
- Put tuning values in `.py` files — no `setdefault`, no inline dicts, no
  arch-conditional constants, no hardcoded fallback configs.
- Mix key schemes: GEMM `M_LEQ_x`/`M_GEQ_y`, MOE `bm…` dispatch keys, conv
  shape keys and MHC `C_`/`M_LEQ_` all belong to their own families.
- Reintroduce a probe, a candidate list, or a cross-backend/cross-arch search
  in `resolve_config_dir()`. A miss is an error with a path in it.
- Add `kpack` to a gfx950 config. Triton's AMD backend deprecates `kpack` on
  CDNA4 — it warns and force-overrides `kpack = 1` there, and the parameter is
  slated for removal. No gfx950 config carries it today; keep it that way.
  gfx942 configs still may. The RDNA trees (gfx1151, gfx1201, gfx1250) do
  carry `kpack` in places — those entries predate the rule, so do not add new
  ones and drop them when you retune the family.

### Not tuning configs

Two AOT code paths build directories under this tree at runtime that are **not
checked in and out of scope**:

- `configs/gemm/aot/<kernel>_M=…-N=…-K=…` —
  `gemm/fused/fused_gemm_afp4wfp4_a16w16.py`,
  `gemm/fused/fused_gemm_afp4wfp4_mul_add.py`
- `configs/paged_mqa_logits/aot/<kernel>` — `attention/pa_mqa_logits.py`

Both hold compiled-kernel metadata, not tuning parameters: the GEMM ones are
written only under `use_aot and os.path.exists(...)`, and the
`paged_mqa_logits` one only on the AOT gluon branch. Do not create, migrate,
or document them as config directories.
