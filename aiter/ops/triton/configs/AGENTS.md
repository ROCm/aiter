# Triton kernel configs

These rules apply to every tuned JSON file in `aiter/ops/triton/configs/`
and to the loaders that read them (`aiter/ops/triton/utils/config_utils.py`
and the `utils/*_config_utils.py` modules). Read them before you add, move,
rename or tune a config file.

## Rules

1. Put tuning values in JSON. Do not put them in Python: no `setdefault`, no
   inline dicts, no arch-specific constants and no fallback configs. If a
   value is missing, fix the JSON.
2. Read configs only through a loader in `utils/`. The loaders use
   `resolve_config_dir()` and `load_config_json()`. Do not use
   `json.load(open(...))`, function-attribute caches or paths built by hand
   (`f"{AITER_TRITON_CONFIGS_PATH}/..."`).
3. Use the one layout below. There are no other config folders and no
   fallback search.
4. Do not put the arch in a file name. The default file is `DEFAULT.json`.
5. Move or rename files in a pure `git mv` commit. Change the contents in the
   next commit.
6. Each family has its own key scheme. GEMM files use only `M_LEQ_<x>` keys
   and `"any"` for M buckets. Do not mix GEMM keys, MOE `bm...` keys, conv
   shape keys and MHC `C_` keys.
7. Put `kpack` only in gfx942 configs. On gfx950, Triton gives a warning and
   sets `kpack = 1`, and Triton will remove the parameter. gfx1250 does not
   support it. On RDNA (gfx11xx, gfx120x), it has no effect.
8. Do not add `.gitkeep` files. Make a `<d_type>/` folder only when you add
   files to it.

## Layout

```text
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

| Part | Values |
| ---- | ------ |
| `<arch>` | `gfx942`, `gfx950`, `gfx1100`, `gfx1101`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx1250` |
| `<backend>` | `triton` or `gluon` |
| `<op>` | `gemm`, `moe`, `conv`, `mhc`, `attention`, `gmm`, `fusions`, `quant` |
| `<d_type>` | `config_name.lower().replace("-", "_")`: `GEMM-AFP4WFP4` → `gemm_afp4wfp4` |

Examples:

```text
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
configs/gfx1250/gluon/moe/a8w4/DEFAULT.json
configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json
```

Two config names must not give the same `<d_type>`: `GEMM-FOO-BAR` and
`GEMM-FOO_BAR` use the same folder.

## Loaders

| Module | Functions |
| ------ | --------- |
| `utils/config_utils.py` | `resolve_config_dir`, `load_config_json`, `select_leq_config` |
| `utils/gemm_config_utils.py` | `get_gemm_config`, `compute_splitk_params`, `add_default_gemm_config_params`, `pick_gemm_num_stages` |
| `utils/moe_config_utils.py` | `get_moe_dispatch` |
| `utils/conv_config_utils.py` | `get_conv_config` and the shape-key helpers |
| `utils/mhc_config_utils.py` | `get_mhc_config`, `get_mhc_post_config` and the other MHC loaders |
| `utils/tuned_config_utils.py` | `get_tuned_kernel_config`, `autotune_configs` |
| `utils/quant_config_utils.py` | `get_quant_config` (Gluon quant) |

Each function has one home. Do not add re-exports or forwarding wrappers.
Attention and GMM kernels read their `DEFAULT.json` with
`resolve_config_dir()` and `load_config_json()` directly. A GMM variant can
have a `"dispatch"` list of rules
(`{"config", "min_K", "min_N", "min_avg_rows_per_group"}`). The first rule
that matches selects the config.

- `resolve_config_dir(op, config_name, backend="triton", arch=None)` builds
  the folder path. It does not search and does not check that files exist.
  It checks each argument and fails with `AssertionError` on a bad value. The
  caller sets `backend`: Gluon code passes `"gluon"`. Use `arch=` only for the
  two allowed fallbacks: MHC uses gfx942 and Triton `fused_clamp_act_mul` uses
  gfx950.
- `load_config_json(fpath, required=True)` raises `FileNotFoundError` with the
  path when a required file is missing. With `required=False` it returns
  `None`. The result is cached per path, also when the file is missing:
  restart the process or call `load_config_json.cache_clear()` after you add a
  file. Copy the returned dict before you change it.
- `select_leq_config(configs, value, prefix="N_LEQ_")` returns a copy of the
  entry with the smallest bound that is not less than `value`, or `any`. For
  more than one axis, use `axes=("M", "N")` and keys such as
  `M_LEQ_32.N_LEQ_1024`.

`tuned_config_utils._get_tuned_kernel_entry()` still builds its path by hand.
Move it to `resolve_config_dir()` when you change it.

## GEMM

```python
get_gemm_config(config_name, M, N=None, K=None, bounds=None,
                specialized_filename=None, backend="triton", B=None)
    -> (config, is_tuned)
```

1. It reads `<arch>/<backend>/gemm/<d_type>/DEFAULT.json`. This file must
   exist.
2. It uses a specialized file if one matches: `<CONFIG_NAME>-B=..-N=..-K=..`,
   then `<CONFIG_NAME>-N=..-K=..`, or `<CONFIG_NAME>-<specialized_filename>`.
   A specialized file with `"DEFAULT_FALLBACK": true` gets the buckets that
   it does not have from `DEFAULT.json`.
3. In the file it tries the `M_LEQ_<x>` keys in increasing order, then
   `"any"`.

`is_tuned` is `True` only when the bucket that matched comes from the
specialized file. It is `False` for `"any"` and for a bucket from
`DEFAULT.json`. Keep it in the return value. The config is a new copy, so you
can change it.

Keep `_get_config()` thin. Set the `backend` default to `"triton"`, never
`None`:

```python
def _get_config(M: int, N: int, K: int, backend: str = "triton"):
    return get_gemm_config("GEMM-A16W16", M, N, K, backend=backend)
```

For split-K kernels, also call `compute_splitk_params(config, K)`.

File format:

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_LEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- The bounds are `STANDARD_M_BOUNDS = (1, 4, 8, ..., 8192)`. A file can set
  its own `M_BOUNDS` list. A `bounds=` argument has priority.
- Include `"any"`, unless the bounds cover all values of M.
- Use only `M_LEQ_<x>` keys and `"any"` for M buckets. Do not use the old
  `"small"` and `"large"` keys.
- `DEFAULT.json` sets the keys of the family: the keys that the kernel reads.
  Each entry has these keys. For most Triton GEMM families they are
  `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`,
  `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`,
  `cache_modifier` and `NUM_KSPLIT`. The tuning scripts tune only the keys
  in `DEFAULT.json`.
- The tuning scripts write a specialized file with `"DEFAULT_FALLBACK": true`,
  the tuned buckets and a copy of the `"any"` of `DEFAULT.json`.
- You can use `cache_modifier` on all archs, also on gfx1250.

Names:

| Kind | Example |
| ---- | ------- |
| Default | `gemm_a16w16/DEFAULT.json` |
| N and K | `gemm_a16w16/GEMM-A16W16-N=256-K=7168.json` |
| Batched | `batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json` |
| Custom suffix | `fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json` |

Name patterns: `GEMM-A{x}W{y}`, `BATCHED_GEMM-A{x}W{y}`, `FUSED-GEMM-{op}`,
`FF-A{x}W{y}-fused`, with the suffixes `_PRESHUFFLED` and `_BLOCKSCALE`.

In AFP4WFP4 file names, `K` is the logical K (`2 * K_bytes`). The wrapper
doubles K before it calls `get_gemm_config()`.

## MOE

`get_moe_dispatch(config_name, arch, backend)` is the only MOE config reader.
It reads `<arch>/<backend>/moe/<d_type>/DEFAULT.json`. It returns `{}` when
there is no file. Do not change the returned dict.

| Family | Backend | Keys | Entry keys |
| ------ | ------- | ---- | ---------- |
| `A8W4` | `triton` | `bm<block_m>_n<N>_k<K>` | `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim` |
| `A8W4` | `gluon` | `bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any` | `block_n`, `block_k`, `num_buffers`, `num_warps`, `persistent_iters` |
| `A4W4` | `gluon` | `bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any` | `block_n`, `block_k`, `num_buffers`, `num_warps` |

- `block_m` is the key. Do not put `BLOCK_SIZE_M` or `block_m` in an entry.
- `<bucket>` comes from `m2bucket()`: `tiny`, `small`, `medium`, `medium2`,
  `large` and `xlarge` (M split at 8, 32, 128, 256 and 512). Give each new
  Gluon shape all six buckets. A missing bucket falls back to
  `bm<block_m>_any`.
- Each Gluon file has a `bm<block_m>_any` entry for each `block_m`.

## Conv

`get_conv_config(config_name, shape_key=None, M=None, variants=())` reads
`<arch>/triton/conv/<d_type>/DEFAULT.json`. It uses the first match of:
`shapes_<variant>[shape_key]`, `shapes[shape_key]`, `M_LEQ_<n>`, `"any"`.
Make shape keys with `format_shape_key()` or `format_prepack_shape_key()`
(and the `_3d` versions). Do not write them by hand.

## MHC

`get_mhc_config(config_name, M, C, mode)` reads `<arch>/triton/mhc/<d_type>/`.
`mode` must be `"sinkhorn"`. `<d_type>` is `f"{config_name}_{mode.upper()}"`
in lowercase (`mhc_fused_sinkhorn`).

- It selects the largest `-C=<value>` file that is not more than C.
- In that file, it selects the largest `M_LEQ_<x>` that is not more than M,
  else `"any"`.
- An arch without an MHC folder uses the gfx942 files. Do not copy this
  fallback to other families.

`get_mhc_post_config(M, C)` reads `mhc_post/DEFAULT.json`.

## Autotune

- Give every `@triton.autotune` its list through
  `autotune_configs(family, configs, default_config=None)`. It returns one
  config, so nothing benchmarks at launch. With `<FAMILY>_TRITON_AUTOTUNE=1`
  it returns all candidates.
- `get_tuned_kernel_config(op, config_name, kernel_name, fallback)` reads the
  pinned tile for the running arch from `DEFAULT.json`. The `fallback` must
  work on all archs.

## Add a config

1. Select the config name. Make sure that its `<d_type>` is not already used.
2. Add `<arch>/<backend>/<op>/<d_type>/DEFAULT.json`.
3. Add specialized files next to it, with the `<CONFIG_NAME>-` prefix, for
   example `GEMM-A16W16-N=256-K=7168.json`. For GEMM, run the tuning scripts
   in `utils/_triton/tuning/`. They write the file into this folder.
4. Move tuning values that are still in Python into the JSON.
5. Test on the target arch. Make sure that the config loads, that `is_tuned`
   is `True` for each tuned M and that the results do not change. For GEMM,
   `verify_configs.py --expect-tuned` checks `is_tuned`.

## Seed a new arch

When an arch has no file for a family, copy the file from the nearest
measured arch. The copy must be byte identical. Say it in the commit message.
The allowed seeds are gfx950 → gfx1250 and gfx1151 → gfx1150, Triton only.
Never seed a Gluon folder. Never seed into gfx950. Replace seeds with
measured values.

## Not configs

`configs/gemm/aot/` and `configs/paged_mqa_logits/aot/` are runtime caches of
compiled kernels. Do not commit them.
