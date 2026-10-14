# AITER Triton ops

This folder has the Triton and Gluon kernels of AITER and their Python
wrappers. Every change in `aiter/ops/triton/` must obey the rules below.
Coding agents find these rules through [`AGENTS.md`](AGENTS.md). Copilot
uses the same rules to review PRs
(`.github/instructions/aiter-ops-triton.instructions.md`). The full config
rules are in [`configs/AGENTS.md`](configs/AGENTS.md).

## Rules

### Kernels

1. Before you add a kernel, find kernels with the same or almost the same
   function. Extend or reuse them. Do not add a duplicate.
2. Put a new kernel in its own PR, together with its wrapper, unit test and
   benchmark.
3. Put the wrapper in the correct category folder. Put the kernel body at the
   same path under `_triton_kernels/` (Triton) or `_gluon_kernels/<arch>/`
   (Gluon). Do not add files at the top of `aiter/ops/triton/`.
4. Launch a kernel only from its public wrapper. Tests, benchmarks and other
   code call the wrapper.
5. Give each launchable kernel a `repr=` from `make_kernel_repr`. See
   [Config-aware kernel names in traces](#config-aware-kernel-names-in-traces).
6. Use the [shared helpers](#shared-helpers). Do not write local copies of
   common code, such as activations, shuffles or config loading. Put a new
   shared helper in the shared module, not in a kernel file. Shared
   `@triton.jit` device helpers go in `_triton_kernels/common/`.
7. Use architecture names (`gfx942`, `gfx950`, `gfx1250`) in code and folder
   names. Do not use product names such as MI300 or MI355.
8. Allocate tensors on the device of the inputs (`device=x.device`). Do not
   use `device="cuda"` in library code.
9. Use absolute imports from the category path, for example
   `from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16`. Do not
   use relative imports or the old flat paths.
10. Do not import `torch` in `utils/_triton/`, in config loading, or in new
    kernel modules. These parts must also work without PyTorch, for example
    with JAX-Triton.

### Configs

11. Put tuning values in the JSON files in `configs/`. Do not put tuning
    values in Python.
12. Read configs only through the loaders in `utils/`.
13. In GEMM configs, use only `M_LEQ_<x>` keys and `"any"` for M buckets.
14. Use `autotune_configs()` for every `@triton.autotune` config list.
15. Put `kpack` only in gfx942 configs.

### Tests and benchmarks

16. Add a unit test in `op_tests/triton_tests/`. Use the folder that matches
    the wrapper folder. See the
    [test rules](../../../op_tests/triton_tests/README.md).
17. Add a benchmark: `op_tests/op_benchmarks/triton/bench_<op>.py`.

### Comments and docs

18. Keep comments short: one or two sentences that tell what the code does,
    or why. Do not write long explanations.
19. Give each public wrapper a docstring: what it computes, its arguments,
    its return value and its limits.
20. Log with `%` placeholders. Library code uses `AiterTritonLogger()`, and
    tests and benchmarks use `from aiter import logger`. Do not use `print`
    or f-strings in log calls.
21. Write docs, comments and PR descriptions in
    [ASD-STE100](https://www.asd-ste100.org/) (Simplified Technical English):
    short sentences, simple words, one instruction in each sentence.
    You can use technical terms.

### Pull requests

22. Change one backend in each PR. Triton and Gluon are one backend.
23. Keep PRs small: one problem in each PR. For work that depends on other
    work, use stacked PRs.
24. Write the PR description with these parts: motivation, technical
    details, test plan and test results.
25. Assign the PR to yourself and answer the reviewers. Code owners can close
    PRs that stay inactive.
26. Make sure that the PR has the `Triton/Gluon` label. Add the
    `ci:triton-300x` label when you change gfx942 code or configs.
27. Review your PR yourself before you ask for a review.
28. Open the PR from the `ROCm/aiter` repository. Use a fork only if you do
    not have access.
29. Remove the `Co-authored-by` lines of AI agents from your commits before
    the merge.

## Reference

### Folder layout

```text
aiter/ops/triton/
├── gemm/              GEMM wrappers: basic/, batched/, feed_forward/, fused/, grouped/
├── attention/         MHA, MLA, lean attention, unified attention
├── moe/               mixture-of-experts ops
├── normalization/     RMSNorm, LayerNorm and fused add + norm
├── quant/             FP8, MXFP4 and MXFP8 quantization
├── rope/              RoPE and fused RoPE ops
├── fusions/           small fused kernels
├── comms/             multi-GPU communication kernels
├── conv/              convolution kernels
├── gated_delta_net/   Gated DeltaNet ops
├── kimi_delta_attn/   Kimi Delta Attention
├── gluon/             Gluon wrappers
├── _triton_kernels/   Triton kernel bodies, same paths as the wrappers;
│                      shared device helpers in common/
├── _gluon_kernels/    Gluon kernel bodies: <arch>/ and the wrapper path
├── configs/           tuned JSON configs
└── utils/             shared code
```

Tests use the same folders under `op_tests/triton_tests/`. Old flat imports
(`aiter.ops.triton.gemm_a16w16`) still work through `_BACKWARD_COMPAT_MAP` in
`__init__.py`. They are only for old external code.

### Shared helpers

| Need | Use |
| ---- | --- |
| Activations in a kernel | `_triton_kernels/common/activation.py` |
| Weight and scale shuffles | `utils/shuffle.py` |
| Config loading | `utils/config_utils.py` and the `utils/*_config_utils.py` loaders |
| Autotune config lists | `utils/tuned_config_utils.py::autotune_configs` |
| Split-K second-stage reduce | `_triton_kernels/common/splitk_reduce.py` |
| Program ID and XCD remapping | `_triton_kernels/common/pid_preprocessing.py` |
| Tile lookup for ragged batches | `_triton_kernels/common/segment_tile.py` |
| Architecture checks | `utils/_triton/arch_info.py::get_arch` |
| Kernel names in traces | `utils/_triton/kernel_repr.py::make_kernel_repr` |
| Logging in library code | `utils/logger.py::AiterTritonLogger`, level from `AITER_TRITON_LOG_LEVEL` |
| Logging in tests and benchmarks | `from aiter import logger`, level from `AITER_LOG_LEVEL` |

### Configs

Each tuned config file is in this path:

```text
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

The suffix depends on the family. For example, GEMM files use
`N=<N>-K=<K>`, batched GEMM files use `B=<B>-N=<N>-K=<K>` and MHC files use
`C=<C>`. Some families use a custom suffix.

Example: `configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=7168.json`.
`<d_type>` is the config name in lowercase, with `_` in place of `-`. A GEMM
kernel loads its config with one call:

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-A16W16", M, N, K)  # returns (config, is_tuned)
```

[`configs/AGENTS.md`](configs/AGENTS.md) has the layout, the file names, the
loaders, the JSON format and the steps to add a config. To tune a GEMM, use
the scripts in [`utils/_triton/tuning/`](utils/_triton/tuning/README.md).
They write the tuned files into `configs/`.

### Config-aware kernel names in traces

`make_kernel_repr` puts the config values in the kernel name. Then each trace
row shows the config that the kernel used.

```python
_gemm_a16w16_repr = make_kernel_repr(
    "_gemm_a16w16_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_KSPLIT"],
)

@triton.jit(repr=_gemm_a16w16_repr)  # Gluon: @gluon.jit(repr=_gemm_a16w16_repr)
def _gemm_a16w16_kernel(...):
    ...
```

Put the compile-time and tuned config keys in the list. A device helper that
you cannot launch alone does not need a `repr`.

### Run tests and benchmarks

```bash
pytest op_tests/triton_tests/              # all tests
pytest op_tests/triton_tests/gemm/basic/   # one folder
python op_tests/op_benchmarks/triton/bench_gemm_a16w16.py
```
