# AITER Triton Ops — Maintainer Guide

This directory contains all Triton- and Gluon-based kernels and their Python
wrappers used by AITER. This README documents the conventions that every
change under `aiter/ops/triton/` is expected to follow. For anything touching
the tuned-config JSON tree, **`configs/CLAUDE.md` is the authoritative
rulebook** — this README summarizes it; that file wins on any conflict.

PR reviews under this directory are additionally checked by Copilot against
`.github/instructions/aiter-ops-triton.instructions.md`, which encodes the
rules below. Any cleanup or refactor that changes a convention must update
both that file and this README in the same PR, so the two stay current.

---

## Directory layout

```text
aiter/ops/triton/
├── __init__.py            # public API + _BACKWARD_COMPAT_MAP (legacy flat imports)
├── gemm/                  # GEMM wrappers: basic/, batched/, feed_forward/, fused/
├── attention/             # MHA, MLA, lean attention, unified attention, ...
├── moe/                   # Mixture-of-experts ops
├── normalization/         # RMSNorm / LayerNorm and fused add+norm variants
├── quant/                 # FP8 / MXFP4 / MXFP8 quantization and fused-quant kernels
├── rope/                  # RoPE and fused QKV-split + RoPE variants
├── fusions/               # small fused glue kernels (mul+add, clamp-act-mul, KV-cache fusions, ...)
├── comms/                 # multi-GPU communication kernels (all-gather, reduce-scatter, comm+compute fusions)
├── conv/                  # convolution kernels (see conv/README.md and conv/DESIGN.md)
├── gated_delta_net/       # Gated DeltaNet ops (gated delta rule, causal conv1d prefill/decode)
├── kimi_delta_attn/       # Kimi Delta Attention (chunked delta attention)
├── gluon/                 # Gluon-backend wrappers
├── _triton_kernels/       # @triton.jit kernel bodies (mirrors the wrapper layout)
├── _gluon_kernels/        # Gluon kernel bodies
├── configs/               # tuned JSON configs — read configs/CLAUDE.md before editing
└── utils/                 # shared machinery: config loading, shuffling, repr, arch info
```

Public wrapper modules live in the categorized folders; the kernel bodies live
in `_triton_kernels/` (and `_gluon_kernels/`) under the same category path.
Tests mirror the same categories under `op_tests/triton_tests/<category>/`.
Kernel bodies are internal: tests, benchmarks, and external code call the
public wrappers only — never `_triton_kernels/` / `_gluon_kernels/` directly.

Legacy flat imports (`from aiter.ops.triton.gemm_a16w16 import ...`) still
resolve through `_BACKWARD_COMPAT_MAP` in `__init__.py`, but **new code must
import from the categorized path** (`aiter.ops.triton.gemm.basic.gemm_a16w16`).

---

## Tuned configs

### Two layouts are live; only one accepts new files

The config tree is mid-migration from a flat, arch-prefixed layout to a nested
one. The legacy flat layout is **deprecated — treat it as read-only history**:

```text
# Target layout (all new GEMM configs go here)
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
#        gfx950   triton    gemm  gemm_afp4wfp4
#                 gluon     moe

# Legacy layout (deprecated, pending removal)
configs/gemm/<arch>-<CONFIG_NAME>[-<suffix>].json
configs/moe/<arch>-MOE-<dtype_str>.json
```

Rules that follow from the layout:

- `<d_type>` is `config_name.lower().replace("-", "_")` (`GEMM-AFP4WFP4` →
  `gemm_afp4wfp4`, see `gemm_config_utils._dtype_dir()`). New config names must
  stay distinct under that fold — `GEMM-FOO-BAR` and `GEMM-FOO_BAR` collide.
- Files inside `configs/<arch>/...` carry **no arch prefix**; the default file
  is named exactly `DEFAULT.json`.
- A `<d_type>/` directory without a `DEFAULT.json` is **invisible** — the
  resolver probes only for the default file, so its specialized files are
  silently ignored. Never split a config family across the two layouts; a
  family migrates wholesale or not at all (`git mv`, 100% rename similarity,
  content changes in a separate commit).
- `<arch>/<backend>/moe/` directories exist but are `.gitkeep` placeholders.
  **MOE has no nested-layout resolver yet** — MOE configs stay in
  `configs/moe/` with the arch prefix until `get_moe_config()` lands
  (design in `configs/CLAUDE.md` §5).
- `kpack` is deprecated starting from gfx950: the Triton AMD backend warns
  and force-overrides `kpack = 1` there, and the parameter is slated for
  removal. Configs for gfx950 and newer must not carry it; only gfx942
  configs may still set it.

### How GEMM configs resolve — `get_gemm_config()`

All GEMM-family kernels load configs through one function,
`utils/gemm_config_utils.py::get_gemm_config(config_name, M, N=None, K=None,
bounds=None, specialized_filename=None, backend=None, B=None)`. It probes
candidate directories for the default file and takes the first hit
(`backend=None` → `<arch>/triton/gemm/` → `<arch>/gluon/gemm/` → legacy
`configs/gemm/`), then reads specialized files from that same directory.
It returns `(config, is_tuned)`:

- the config is a fresh deep copy, safe to mutate;
- `is_tuned` is `True` only when a specialized (`N=…-K=…`, `B=…-N=…-K=…`, or
  `specialized_filename`) file was hit. `_get_config()` passes the pair
  through unchanged; the flag is there so callers and tuning tooling can
  detect shapes running on untuned defaults (call sites that don't need it
  may ignore it).

The per-kernel `_get_config()` must stay a thin wrapper:

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-A16W16", M, N, K)

# Split-K kernels:
def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return compute_splitk_params(config, K), is_tuned
```

Split-K kernels also share one common second-stage reduce —
`_gemm_splitk_reduce_kernel` (and `_batched_gemm_splitk_reduce_kernel`) in
`_triton_kernels/common/splitk_reduce.py` — rather than carrying a per-kernel
reduce stage. New split-K kernels import it from there.

### Config JSON format

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- `M_LEQ_x` keys are searched ascending over
  `STANDARD_M_BOUNDS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)`,
  then `M_GEQ_x` descending, then `any`. Custom `bounds=(...)` must be strictly
  increasing positive ints.
- `any` must exist unless every reachable `M` hits an explicit bound — a
  `KeyError` at lookup time usually means it's missing.
- The old `{"small": ..., "large": ...}` shape is **banned**.
- Each `M_*` entry carries at minimum: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
  `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`,
  `matrix_instr_nonkdim`, `cache_modifier`, `NUM_KSPLIT`.
  (`add_default_gemm_config_params()` backfills `NUM_KSPLIT`/`cache_modifier`
  as a last resort; that is not a license to omit keys.)

### Tuning values live in JSON, never in Python

No `setdefault(...)` blocks, no inline config dict literals, no
arch-conditional constants, no hardcoded fallback configs in `.py` files. If a
value is missing at runtime, the fix is in the JSON. MOE still has legacy
Python fallbacks (`get_optimal_moe_config()`, the `moe_op_gemm_a8w4` dispatch
tiers) — fix them as you touch them, and do not add more.

### Loading is unified — `load_config_json()`

Every config-file read goes through `utils/core.py::load_config_json(fpath,
required=...)`. It is cached per path **including negative results**:
a config file added at runtime is not picked up without
`load_config_json.cache_clear()` or a process restart. Do not hand-roll
`json.load(open(...))` or function-attribute caches, and prefer the resolvers
(`get_gemm_config` / `get_tuned_kernel_config`) over hand-built
`f"{AITER_TRITON_CONFIGS_PATH}/..."` paths — hand-built paths break silently
when a family migrates and must be grepped for during every migration.

Kernels that carry a Python autotune search space (opt-in tuning) pin their
single default tile per arch via
`utils/tuned_config_utils.py::get_tuned_kernel_config(op, config_name,
kernel_name, fallback, backend)`, which reads the nested-layout
`DEFAULT.json`. The `fallback` must be launchable on any arch, not fast on one.

### Config naming

| Kind             | Pattern                                                        |
| ---------------- | -------------------------------------------------------------- |
| Basic GEMM       | `GEMM-A{x}W{y}` (+ variants `_BLOCKSCALE`, `_PRESHUFFLED`, ...) |
| Batched GEMM     | `BATCHED_GEMM-A{x}W{y}`, specialized `-B={B}-N={N}-K={K}`       |
| Fused ops        | `FUSED-GEMM-{operation}`                                        |
| Feed-forward     | `FF-A{x}W{y}-fused`                                             |
| MOE              | `MOE-<dtype_str>` (`DEFAULT`, `FP8_W8A8`, `MX_FP4`, ...)        |

- **`K` in AFP4WFP4 filenames is the logical K, i.e. `2 * K_bytes`** — the
  wrapper doubles K before calling `get_gemm_config`. Tuning output named by
  the packed byte width will never be found.
- MOE files use `small_M` / `medium_M` / `large_M` (thresholds 256 / 1024 in
  `moe_config_utils.py`). Never mix that scheme with the GEMM
  `M_LEQ_x`/`M_GEQ_y` scheme in either direction.
- `configs/gemm/aot/` and `configs/paged_mqa_logits/aot/` are runtime AOT
  caches, not tuning configs — never check them in or migrate them.

For migrating a family into the nested layout, follow the playbook in
`configs/CLAUDE.md` §6 step by step. For the manual tuning flow, see
`utils/_triton/tunning/README.md`.

---

## Config-aware kernel names in traces (`kernel_repr`)

Kernel names in traces embed the compile-time config so a trace row can be
matched to the exact tuned config
(`utils/_triton/kernel_repr.py::make_kernel_repr`):

```python
_gemm_a16w16_repr = make_kernel_repr(
    "_gemm_a16w16_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_KSPLIT"],
)

@triton.jit(repr=_gemm_a16w16_repr)
def _gemm_a16w16_kernel(...):
    ...
```

Values are read from `specialization.constants` and sanitized (`None → NONE`,
bools → `0/1`, strings uppercased with non-alphanumerics folded to `_`).
**Every new kernel gets a `repr=`** with its meaningful config keys.

---

## Weight & scale shuffling — one home: `utils/shuffle.py`

All pre-shuffle/layout-permute helpers for weights and scales were unified
into `aiter/ops/triton/utils/shuffle.py`. Kernel
wrappers, Gluon paths, tests, and benchmarks import from there — the
per-kernel copies that used to live in `moe_op_gemm_*.py` and test files were
deleted. **Do not re-implement a shuffle** (the telltale is a local
`view → permute → contiguous` helper); if a new layout is needed, add it here.

| Function                 | Purpose                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| `shuffle_weight(x, ...)` | Arch-aware weight preshuffle: gfx1250 WMMA/TDM path, otherwise delegates to `aiter.ops.shuffle.shuffle_weight` |
| `moe_weight_decode_view(w)` | Zero-copy `(E, N, K)` → decode view sharing storage (renamed from `moe_weight_gfx1250_decode_view`) |
| `shuffle_scale_gemm` / `unshuffle_scale_gemm` | GEMM MX-scale tiles — gfx950 `(32, 8)`, gfx1250 `(16, 4)` |
| `shuffle_scale_moe`      | MoE MX scales (a8w4/a8w8/a16w4/a4w4); `return_layout=True` also returns the `SWIZZLE_MX_SCALE` label (`CDNA4_SCALE`/`GFX1250_SCALE`); no-op on arches without a native layout (e.g. gfx942) |
| `shuffle_scale_batched`  | FP4 blockscale16 batched scales, arch-independent                        |

Callers should stay arch-agnostic: pass `arch=None` and let `get_arch()`
decide, and use `return_layout=True` instead of hardcoding swizzle labels.

---

## Architecture naming

Key behavior off GPU architecture identifiers, never product names — in config
filenames (`gfx950-...`), directory names (`configs/gfx950/...`), and code:

```python
DEVICE_ARCH = arch_info.get_arch()

if DEVICE_ARCH in ("gfx950", "gfx1250"):   # correct
    ...
# Never: parsing "MI300"/"MI350" product strings or comparing chip numbers.
```

---

## Docstrings

Every public wrapper carries a docstring stating what the kernel computes, the
args (including which config parameters apply and what they control), the
return value, and any special considerations (unsupported options, layout
expectations such as "weights must be pre-shuffled", etc.).

---

## Tests

Tests live under `op_tests/triton_tests/<category>/`, mirroring this
directory's categories:

```bash
pytest op_tests/triton_tests/              # everything
pytest op_tests/triton_tests/gemm/basic/   # one subset
```

---

## Checklist for a new kernel

- Reuse first: check `utils/`, `_triton_kernels/common/`, and existing
  kernels before writing new helpers — don't duplicate code that already
  exists in the tree.
- Wrapper in the right category folder; kernel body under `_triton_kernels/`
  (or `_gluon_kernels/`) at the same path, launched only through the wrapper.
  No new top-level flat files.
- `_get_config()` is a thin `get_gemm_config(...)` call; split-K goes through
  `compute_splitk_params()`. No tuning values in Python.
- Config JSON in the **nested layout** (`configs/<arch>/<backend>/<op>/<d_type>/`),
  `M_LEQ/M_GEQ/any` keys, all required params present.
- `make_kernel_repr(...)` + `@triton.jit(repr=...)`.
- Weight/scale shuffling imported from `utils/shuffle.py`.
- Arch checks against `gfx*` identifiers.
- Wrapper docstring.
- Unit test under `op_tests/triton_tests/<category>/` and a benchmark script
  under `op_tests/op_benchmarks/triton/bench_<op>.py` — a kernel ships as
  kernel + wrapper + test + benchmark.
