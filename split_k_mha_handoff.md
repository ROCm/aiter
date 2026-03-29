# split_k_mha handoff

## Current branch

- Branch: `split_k_mha`
- Remote: `origin/split_k_mha`
- Pushed commit: `165ac40cc`
- Commit title: `add MHA prefill split-K metadata scaffold`

## Goal

Add a new metadata path for MHA prefill split-K work generation, without using the current persistent scheduler, while keeping the output layout compatible with persistent-style `work_info_set + reduce_*`.

Target scenario:

- prefill
- `uni_seqlen_qo >= 128`
- user-specified `split_k_size`
- persistent-style metadata layout
- non-persistent execution model

## What was added

### Python

- `aiter/ops/attention.py`
  - `get_mha_prefill_splitk_metadata_info_v1(...)`
  - `get_mha_prefill_splitk_metadata_v1(...)`

### C++ / pybind / JIT

- `csrc/include/mha_prefill_splitk_metadata.h`
- `csrc/kernels/mha/prefill_splitk_metadata.cu`
- `csrc/pybind/mha_prefill_splitk_metadata_pybind.cu`
- `csrc/include/rocm_ops.hpp`
  - added `MHA_PREFILL_SPLITK_METADATA_PYBIND`
- `aiter/jit/optCompilerConfig.json`
  - added `module_mha_prefill_splitk_metadata`

### Tests

- `op_tests/test_mha_prefill_splitk_metadata.py`

## Current implementation status

This is a scaffold / first working cut, not a final kernel implementation.

Implemented:

- Python API and JIT module wiring
- pybind export
- host-side metadata generation in `csrc/kernels/mha/prefill_splitk_metadata.cu`
- test skeleton for metadata shape/content validation
- branch created and pushed

Not yet done:

- actual build validation of `module_mha_prefill_splitk_metadata`
- runtime test execution
- stage1 kernel integration
- reduce integration verification
- possible device/kernel-side metadata generation rewrite

## Important design choices

- This path does **not** use the current MLA persistent scheduler.
- Metadata is generated in a simplified flat form.
- `work_indptr` is simplified to `[0, num_work]`.
- `kv_offset` is currently filled with `0`.
- `partial_qo_loc` currently advances by tile token length.

## Expected metadata semantics

Each work item corresponds to:

- `(batch_idx, qo_tile, kv_split)`

Each reduce group corresponds to:

- `(batch_idx, qo_tile)`

Causal rule currently used:

- `allowed_kv_end = min(kv_begin + (qo_end_tile - qo_begin), kv_end_total)`

Non-causal rule currently used:

- `allowed_kv_end = kv_end_total`

## Files to inspect first when resuming

1. `split_k_mha_handoff.md`
2. `aiter/ops/attention.py`
3. `csrc/include/mha_prefill_splitk_metadata.h`
4. `csrc/kernels/mha/prefill_splitk_metadata.cu`
5. `op_tests/test_mha_prefill_splitk_metadata.py`

## Known unrelated workspace state

These were intentionally not included in the split-K MHA commit:

- modified: `3rdparty/composable_kernel`
- untracked: `.cursor/`

Be careful not to accidentally include them in later commits.

## Recommended next steps

1. Build or trigger JIT compilation of `module_mha_prefill_splitk_metadata`.
2. Run metadata-only tests first.
3. Validate `work_info_set`, `reduce_indptr`, `reduce_final_map`, and `reduce_partial_map`.
4. Decide whether `partial_qo_loc` should remain token-offset based or switch to tile-id based.
5. Only after metadata is stable, wire in stage1 execution.

## How to resume with the assistant

After logging into the server:

1. `git checkout split_k_mha`
2. `git pull`
3. open and read `split_k_mha_handoff.md`
4. tell the assistant:
   `继续 split_k_mha_handoff.md 的工作，从 build/validation 开始`

## Short status summary

The work is saved in git plus this handoff note. There is no special "load" command required if the repo is up to date; reading this file is enough to restore the working context quickly.
