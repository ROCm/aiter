---
applyTo: "aiter/ops/triton/**,op_tests/triton_tests/**,op_tests/op_benchmarks/triton/**"
---

# AITER Triton ops: review rules

Flag each change that breaks a rule below. Tell the author which rule it
breaks and how to fix it. The author can be new to these rules.

## How to write review comments

- Write in ASD-STE100 (Simplified Technical English): short sentences, simple
  words, one point in each sentence. You can use technical terms.
- Keep each comment short: one sentence for the problem and one for the fix.
- Do not repeat what the code does. Do not add long explanations.

## PR scope

- A new kernel goes in its own PR, with its wrapper, unit test and benchmark.
  Flag a PR that adds a new kernel and also unrelated work: a refactor, a
  cleanup, a retune or a fix to a different kernel.
- Each PR changes one backend. The backends are Triton/Gluon
  (`aiter/ops/triton/`, `aiter/aot/triton/`), HIP (`csrc/`), ASM (`hsa/`,
  `*_asm.py`), CK (`csrc/ck_*`, `ck_tile`), OPUS (`aiter/ops/opus/`) and FlyDSL
  (`aiter/ops/flydsl/`, `aiter/aot/flydsl/`). Triton and Gluon are one
  backend. List the files of the other backend. When the parts depend on each
  other, suggest stacked PRs. Tests and benchmarks belong to the backend that
  they test. Docs, CI files and other wrappers are not a backend.
- Each PR fixes one problem. Flag a PR that fixes two or more independent
  problems, for example a bug fix and a refactor. Name the part to move to a
  follow-up PR.

## Reuse

- Flag new code that duplicates existing code, also in part. The fix is to
  extend or import the existing code. Look in `utils/`,
  `_triton_kernels/common/` (activations, program ID remapping, split-K
  reduce, ragged-batch tile lookup), the existing kernels and the test
  helpers.
- Flag a function with a second home: a re-export, a wrapper that only
  forwards the call, or a copy of a core helper in a family loader.

## Folders and imports

- Flag a new module at the top of `aiter/ops/triton/`. A wrapper goes in its
  category folder.
- Flag a launchable `@triton.jit` or `@gluon.jit` kernel in a public wrapper
  module. Triton kernel bodies go in `_triton_kernels/` and Gluon kernel
  bodies go in `_gluon_kernels/<arch>/`, at the path of the wrapper. A device
  helper that only another kernel calls can stay with that kernel.
- Flag a kernel path under `_triton_kernels/` or `_gluon_kernels/<arch>/` that
  is different from the folder of its wrapper. The kernel folder has the name
  of the wrapper folder: `normalization/`, not `norm/`. Do not flag the old
  `_triton_kernels/flash_attn_triton_amd/` folder (used by
  `attention/mha.py`). A kernel that wrappers in several folders use stays
  with the wrapper that owns it, or goes in `_triton_kernels/common/`.
- Flag a new `@triton.jit` device helper that kernels in several folders use,
  when it is not in `_triton_kernels/common/`. Examples:
  `common/activation.py`, `common/pid_preprocessing.py`. Do not put such a
  helper in `utils/_triton/` or in the kernel module of one category.
- Flag a PR that moves a wrapper but not its kernels, or that does not update
  all importers of the old path.
- Flag common helpers (config loading, shuffles, arch detection, logging)
  that a kernel file writes again instead of importing them from `utils/`.
- Flag imports through the old flat paths
  (`from aiter.ops.triton.gemm_a16w16 import ...`). Use the category path.
- Flag relative imports. Use absolute imports
  (`from aiter.ops.triton.<...> import ...`).

## No torch in torch-free code

These parts must work without PyTorch, for example with JAX-Triton. Flag:

- `torch` in a module under `utils/_triton/`. Exception:
  `utils/_triton/tuning/`. Put the torch part in `utils/`.
- `torch` in config loading: `utils/config_utils.py` and
  `utils/*_config_utils.py`.
- `torch` in a new kernel module, or a first `torch` import in a kernel module
  that has none. Put host code (tensor type hints, dtypes, `torch.empty`) in
  the wrapper.

## Configs: files and names

All tuned configs are in `configs/<arch>/<backend>/<op>/<d_type>/`. `<op>` is
`gemm`, `moe`, `conv`, `mhc`, `attention`, `gmm`, `fusions` or `quant`.
`<d_type>` is `config_name.lower().replace("-", "_")`. Flag:

- A config file outside this layout, for example in `configs/gemm/` or at the
  top of `configs/`.
- An arch prefix in a file name in `configs/<arch>/...`. The default file must
  be `DEFAULT.json`.
- A specialized file in a folder without `DEFAULT.json`, when the loader needs
  the default.
- Two folders for one family that differ only by `-` and `_`
  (`GEMM-FOO-BAR` and `GEMM-FOO_BAR`).
- A config file that is moved and changed in the same commit. Use a pure
  `git mv` commit, then a second commit for the change.
- A new `.gitkeep` file under `configs/`.
- A new arch folder copied from another arch, when the copy is not byte
  identical or the commit message does not say so. The allowed copies are
  gfx950 → gfx1250 and gfx1151 → gfx1150, Triton only. Never copy into a Gluon
  folder or into gfx950.
- `kpack` in a config outside gfx942.
- Files under `configs/gemm/aot/` or `configs/paged_mqa_logits/aot/`. These are
  runtime caches. Do not commit them.

## Configs: JSON contents

In GEMM configs, flag:

- An M bucket key that is not `M_LEQ_<x>` or `"any"`, such as `"small"` or
  `"large"`. Use only `M_LEQ_<x>` keys and `"any"` for M buckets.
- A new file without `"any"`, when the bounds do not cover all values of M
  and the file does not have `"DEFAULT_FALLBACK": true`.
- An entry without one of the keys of the family's `DEFAULT.json`.
- A `DEFAULT.json` without a key that its kernel reads. The tuning scripts
  tune only the keys in `DEFAULT.json`. For most Triton GEMM families the keys
  are `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`,
  `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`,
  `cache_modifier`, `NUM_KSPLIT`.
- MOE keys (`bm<block_m>_n<N>_k<K>`) in a GEMM config, or GEMM keys in a MOE
  table.
- An `*AFP4WFP4*` file name with the packed K. The file name uses the logical
  K (`2 * K_bytes`).

Do not flag:

- A `cache_modifier` in a gfx1250 config. You can use cache modifiers on all
  archs, also on gfx1250.
- A specialized file with `"DEFAULT_FALLBACK": true` that has only some
  buckets. The loader takes the other buckets from `DEFAULT.json`.

In MOE dispatch tables, flag:

- A new Gluon shape without all six buckets: `tiny`, `small`, `medium`,
  `medium2`, `large`, `xlarge`.
- A Gluon file without a `bm<block_m>_any` entry for a `block_m` that it uses.
- `BLOCK_SIZE_M` or `block_m` in an entry. `block_m` is the key, not a tuned
  value.
- Triton keys (`BLOCK_SIZE_N`, `num_stages`) in a Gluon file, or Gluon keys
  (`block_n`, `num_buffers`, `persistent_iters`) in a Triton file.

## Configs: Python code

Tuning values are in JSON, never in Python. This applies to Triton and Gluon.
Flag:

- Tuning values in `.py` files: `config.setdefault(...)`, dict literals with
  `BLOCK_SIZE_*`, `num_warps` or `waves_per_eu`, arch-specific tuning
  constants, or fallback configs.
- A kernel `_get_config()` with a `backend` argument that defaults to `None`.
  The default must be `"triton"`. A public wrapper with `backend=None` must set
  the backend before it calls down.
- A GEMM `_get_config()` that does more than call `get_gemm_config(...)` (and
  `compute_splitk_params()` for split-K), or that does not return
  `(config, is_tuned)`:

  ```python
  def _get_config(M: int, N: int, K: int):
      return get_gemm_config("GEMM-A16W16", M, N, K)
  ```

- `json.load(open(...))` or a function-attribute cache. Use
  `load_config_json()` or a family loader.
- A change to the dict that `load_config_json()` returns. It is a shared
  cached object. Copy it first.
- A config path built by hand (`f"{AITER_TRITON_CONFIGS_PATH}/..."`). Use
  `resolve_config_dir()` or a family loader.
- A loop that finds the smallest `N_LEQ_*` entry, or a bucket walk on more
  than one axis. Use `select_leq_config()`.
- A second MOE config reader. `utils/moe_config_utils.py::get_moe_dispatch` is
  the only one.
- A new arch or backend fallback chain in a loader. The only allowed fallbacks
  are MHC (gfx942) and Triton `fused_clamp_act_mul` (gfx950). Both use the
  `arch=` argument.
- A raw config list in `@triton.autotune`. Use `autotune_configs()`:

  ```python
  @triton.autotune(
      configs=autotune_configs("MY_FAMILY", _get_autotune_configs()),
      key=[...],
  )
  ```

  A raw list benchmarks again for each new key. This costs compile time,
  breaks CUDA-graph capture and makes test results change from run to run.
  Use `default_config=` when the first entry is not the default. A family with
  its own environment variable uses `env=` and `default=`.

## Shuffles

All weight and scale shuffles are in `utils/shuffle.py`. Flag:

- A local shuffle in a wrapper, kernel, test or benchmark. Look for a small
  helper that does `view` → `permute` → `contiguous` on weights or scales.
- The old name `moe_weight_gfx1250_decode_view`. Use `moe_weight_decode_view`.
- A hard-coded `SWIZZLE_MX_SCALE` label (`"CDNA4_SCALE"`, `"GFX1250_SCALE"`).
  Use `shuffle_scale_moe(..., return_layout=True)`.

## Kernels

- Flag a launchable `@triton.jit` or `@gluon.jit` kernel without `repr=`. Use
  `make_kernel_repr` from `aiter.ops.triton.utils._triton.kernel_repr`, with
  the compile-time and tuned config keys:

  ```python
  _kernel_repr = make_kernel_repr("_kernel_name", ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "num_warps"])

  @triton.jit(repr=_kernel_repr)  # Gluon: @gluon.jit(repr=_kernel_repr)
  ```

  A device helper that you cannot launch alone does not need a `repr`.
- Flag a split-K kernel with its own reduce kernel. Use
  `_gemm_splitk_reduce_kernel` or `_batched_gemm_splitk_reduce_kernel` from
  `_triton_kernels/common/splitk_reduce.py`.
- Flag a kernel that tests, benchmarks or other code import or launch
  directly. Only its wrapper launches it. Flag a new kernel without a public
  wrapper.
- Flag product names (`MI300`, `MI350`, `MI355`) in names, file names or
  logic. Compare arch names:

  ```python
  if DEVICE_ARCH in ("gfx950", "gfx1250"): ...
  ```

- Flag `device="cuda"`, `torch.device("cuda")` and `.cuda()` in wrappers and
  launch code. Allocate on the device of the input:
  `torch.empty(shape, dtype=x.dtype, device=x.device)`. Docstring examples and
  tests can use `"cuda"`.
- Flag a new public wrapper without a docstring. The docstring tells what the
  kernel computes, the arguments, the return value and the limits.
- Flag `print(...)` or a new logger in library code. Use
  `_LOGGER = AiterTritonLogger()` from `aiter.ops.triton.utils.logger`, with
  `%` placeholders. Tests and benchmarks use `from aiter import logger`.

## Comments and docs

- Flag long comments and docstrings. A comment has one or two sentences. It
  tells what the code does, or why. Flag text that repeats the code, explains
  Triton basics or tells what the next line does. Flag commented-out code.
- Flag docs, comments and PR descriptions that do not follow ASD-STE100: long
  sentences, complex grammar or uncommon non-technical words.

## Tests and benchmarks

- Flag a new kernel without a unit test in `op_tests/triton_tests/<category>/`
  or without a benchmark.
- Flag a test that is not a pytest `test_<op>.py` file in its category folder.
  Shared test helpers go in the existing `*_test_utils.py` and `utils/`
  modules.
- The test folder is the folder of the wrapper that the test imports:
  - `attention/mla.py` → `op_tests/triton_tests/attention/test_mla.py`
  - `gemm/basic/gemm_a8w8.py` → `gemm/basic/test_gemm_a8w8.py`
  - `moe/moe_routing/routing.py` → `moe/moe_routing/`
  - `attention/mha.py` → `attention/`, also if its kernels are in
    `flash_attn_triton_amd/`
  - flat wrappers (`topk.py`, `softmax.py`) → the `triton_tests/` root
  - `utils/` helpers → `triton_tests/utils/`

  `torch_compile/` and `triton_metadata_redirect/` are infrastructure suites.
  Each op test folder has an `__init__.py`. Flag:

  - A test in a different folder than the wrapper that it imports.
  - A new test folder without a matching folder in `aiter/ops/triton/`.
  - A moved wrapper without its kernels and test, or a moved test with
    importers of the old path.
- Flag configs in tests: dicts with `BLOCK_SIZE_*`, `num_warps` or
  `waves_per_eu`, or a literal `config=` argument. Tests use the config that
  the wrapper loads.
- Flag `print(...)` of tensors, shapes or timings, and
  `if __name__ == "__main__"` report blocks. Tests check results with asserts
  (`torch.testing.assert_close`). Do not flag the old prints in
  `gemm/basic/test_gemm_a8wfp4.py`, `quant/test_quant.py` and
  `conv/_helpers.py`.
- Flag f-strings and `+` in log calls. Use `%` placeholders:
  `logger.info("shape=%s", x.shape)`. Use `%d` and `%f` only for numbers that
  are never `None`. Use `%s` for all other values.
- Flag failure detail at INFO level. Pytest sets `AITER_LOG_LEVEL=WARNING`, so
  CI does not show INFO. Use WARNING or higher, or put the detail in the
  assert message.
- Flag a test that logs pass or fail but does not assert. That test cannot
  fail.
- Flag `if DEBUG_MODE: logger.info(...)` and new `DEBUG_MODE` flags. Use
  `logger.debug(...)` and `AITER_LOG_LEVEL=DEBUG`. Flag `logger.setLevel(...)`
  and `logging.basicConfig(...)` in tests.
- Flag `@triton.autotune`, config sweeps and tile-size loops in tests. Tuning
  belongs in the tuning scripts and benchmarks.
- Benchmarks are `op_tests/op_benchmarks/triton/bench_<op>.py` and look like
  the existing files. The config and shuffle rules also apply to them.

## Keep the docs current

A PR that changes a convention under `aiter/ops/triton/` must update
`aiter/ops/triton/README.md` and this file. Examples: a moved shared helper, a
change to the config layout or the loaders, or new folders. Flag a PR that
does not.

## Agent instruction files

Agent instructions go in `AGENTS.md` files, so that all coding agents read
them. `aiter/ops/triton/CLAUDE.md` is the only `CLAUDE.md` file. It is
correct and must stay: it imports the `AGENTS.md` files for Claude Code. Do
not flag it and do not ask to remove it.

Flag:

- A `CLAUDE.md` file in a different folder.
- Coding rules in `aiter/ops/triton/CLAUDE.md`. It only has a short note and
  the imports. Rules go in an `AGENTS.md` file or in a README.
- A new `AGENTS.md` file without an import in `aiter/ops/triton/CLAUDE.md`.
