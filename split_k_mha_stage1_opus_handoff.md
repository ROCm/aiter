# split_k_mha stage1 opus handoff

## Scope

This note captures the current design direction for the split-K MHA stage1 kernel written in an `opus` style, so work can continue on another machine without reconstructing the context.

This note is focused on the split-K MHA stage1 kernel and its current JIT / Python smoke-test status.

## Current branch state

- Branch: `split_k_mha`
- Metadata source: `csrc/kernels/mha/prefill_splitk_metadata.cu`
- Metadata tests: `op_tests/test_mha_prefill_splitk_metadata.py`
- New skeleton file: `csrc/kernels/mha/prefill_splitk_stage1_opus_skeleton.cu`

## Current metadata semantics

The active metadata contract is:

- `work_info.batch_idx`: real batch index only
- `work_info.qo_start/qo_end`: Q token range for this work
- `work_info.kv_start/kv_end`: KV token range for this work
- `work_info.partial_qo_loc`: token-offset into partial output buffer
- `work_info.kv_offset`: computed in MLA style as `kv_end_total - kv_end`

Important: head information is **not** folded into `batch_idx` or `work_info`.

## Current prefix-causal expectation

The current active causal test helper in `op_tests/test_mha_prefill_splitk_metadata.py` validates a prefix-aware causal interpretation:

- `prefix_len = kv_len - q_len`
- `visible_kv_len = min(prefix_len + (q_tile_idx + 1) * q_tile_size, kv_len)`

There are still commented older test blocks in the same file. Do not assume the commented blocks are the active contract.

## Agreed stage1 simplifications

The current design intentionally simplifies stage1 to make bring-up easier:

1. `Q`, `K`, and `V` come from **contiguous buffers**
2. Page-table style arguments remain in the kernel signature for future compatibility
3. `KV page size` is treated as `1`
4. One workgroup handles:
   - one `work_info`
   - one `head`
5. `blockIdx.x -> work_id`
6. `blockIdx.y -> head_idx`
7. `4 waves` are stacked vertically along the `M` dimension

## Agreed launch / tile mapping

Current target mapping:

- `QTileSize = 128`
- `blockDim.x = 256`
- `4` waves per block
- each wave handles `32` rows in `M`
- `N` is processed by looping over the current work's `kv_start:kv_end` range in `NTile` chunks

Wave-to-row mapping:

- wave 0 -> rows `[0, 32)`
- wave 1 -> rows `[32, 64)`
- wave 2 -> rows `[64, 96)`
- wave 3 -> rows `[96, 128)`

## Current stage1 files

The stage1-opus path is currently spread across these files:

- `csrc/kernels/mha/prefill_splitk_stage1_opus_skeleton.cu`
- `csrc/include/mha_prefill_splitk_stage1_opus.h`
- `csrc/pybind/mha_prefill_splitk_stage1_opus_pybind.cu`
- `aiter/ops/attention.py`
- `aiter/jit/optCompilerConfig.json`
- `op_tests/test_mha_prefill_splitk_stage1_opus.py`

## Why the skeleton file exists

`csrc/kernels/mha/prefill_splitk_stage1_opus_skeleton.cu` started as a design skeleton.

It exists to freeze:

- the launch contract
- the `work_info -> block` mapping
- the contiguous `Q/K/V` assumption
- the planned `opus` integration points

It still avoids losing design progress while the actual implementation is changing, but it is no longer purely design-only.

## What is already in the skeleton

The skeleton file already contains:

- `WorkInfoView`
- contiguous `Q/K/V` view helpers
- the simplified kernel signature
- `grid.x / grid.y` work/head mapping
- `opus::make_gmem` and `opus::make_smem` usage for loader structure
- shared-memory staging for `K/V`
- a reference-style scalar `softmax` loop
- a `32x32x8` `opus::make_mfma` draft for `QK`
- a scalar fallback path for `QK`
- a `32x32x8` `opus::make_mfma` draft for `PV`
- a scalar fallback path for `PV`
- an optional debug score dump hook controlled by compile-time macro
- a torch wrapper callable from Python
- a dedicated JIT module name: `module_mha_prefill_splitk_stage1_opus`

## Agreed next compute step

The next implementation step is **not** more metadata work.

The next step is:

1. keep the skeleton launch / loader structure
2. validate the `QK` `32x32x8` MFMA path / score scatter
3. validate the `PV` `32x32x8` MFMA path against a reference implementation

Recommended MFMA choice for first pass:

- `32x32x8`

Recommended decomposition for one wave:

- `Q stripe`: `[32, 128]`
- `K tile`: `[32, 128]`
- `Score`: `[32, 32]`
- a single wave computes one full `32 x 32` score tile
- loop along `K=128` in `8`-wide MFMA steps

For `PV`:

- `P tile`: `[32, 32]`
- `V tile`: `[32, 128]`
- `O tile`: `[32, 128]`
- use a logical transpose view of `V` for MFMA-B

## Current QK implementation note

The current skeleton now contains a `32x32x8` `QK` path using:

- `make_tiled_mma<..., seq<1,1,1>, seq<1,1,1>, seq<32,32,8>, mfma_adaptor_swap_ab>`
- `partition_layout_a / b / c`
- `gmem` for `Q`
- `smem` for `K`
- `partition_layout_c + smem.store` to scatter `score_frag` into a wave-local `[32, 32]` score tile in shared memory

It also materializes the wave-local score tile into shared memory before feeding
the existing scalar online-softmax path.

## Current PV implementation note

The current skeleton now also contains a `32x32x8` `PV` draft:

- softmax probabilities are first materialized to shared memory as a `[32, 32]` tile
- `P` is used as MFMA-A
- `V` is viewed logically as `V^T` for MFMA-B using a transposed stride view
- `O` is accumulated in `32x32` output-dim tiles and scattered back through shared memory

This path is still a draft and mainly proves that the design compiles and launches.

Important:

- this is still a **draft**
- the `QK` MFMA path is callable from Python through JIT
- the module has been compile-verified by smoke test
- the `PV` MFMA draft also compiles and launches through the same smoke path

## Current smoke-test status

The stage1-opus path is currently connected to Python as:

- `aiter.mha_prefill_splitk_stage1_opus(...)`

Current minimal smoke test:

- file: `op_tests/test_mha_prefill_splitk_stage1_opus.py`
- scope: bf16 inputs, `num_heads=1`, `num_kv_heads=1`, `head_dim=128`
- status: passing

The smoke test proves:

- JIT build succeeds
- pybind exposure succeeds
- Python dispatch succeeds
- the kernel launches successfully
- output tensors contain finite values

The smoke test does **not** prove:

- the MFMA score scatter is numerically correct
- the `PV` MFMA path matches a reference attention implementation
- the implementation is generic beyond the constrained shape/dtype path above

## Current known issue from reference test

A reference comparison test now exists:

- `op_tests/test_mha_prefill_splitk_stage1_opus.py::test_mha_prefill_splitk_stage1_opus_matches_reference_single_work`

Current status:

- this test fails
- the failure appears before meaningful `PV` validation
- `debug_qk_scores[0, 0, 0]` is all zeros while the Python reference `Q @ K^T * scale` is non-zero

This strongly suggests the next debugging focus should be on:

1. `partition_layout_a / partition_layout_b` correctness for the `32x32x8` QK path
2. `score_frag -> wave_score_base` scatter via `partition_layout_c`
3. whether the current `make_tiled_mma(..., mfma_adaptor_swap_ab{})` choice matches the intended A/B interpretation

Do not assume `PV` is the first broken stage until `debug_qk_scores` matches the reference tile.

## Current debug hook

There are two compile-time switches at the top of the skeleton:

- `AITER_SPLITK_STAGE1_USE_MFMA_QK`
- `AITER_SPLITK_STAGE1_DEBUG_SCORE_DUMP`

When `AITER_SPLITK_STAGE1_DEBUG_SCORE_DUMP` is enabled and a non-null
`debug_qk_scores` pointer is passed, the kernel writes the wave-local `32x32`
score tile to a debug buffer shaped logically like:

- `[num_work, num_heads, 4, 32, 32]`

This is intended for validating the `score_frag -> score tile` mapping before
replacing the softmax path.

## Important caution

This repository currently has local modifications in these files:

- `aiter/ops/attention.py`
- `csrc/kernels/mha/prefill_splitk_metadata.cu`
- `op_tests/test_mha_prefill_splitk_metadata.py`

Do not assume they match the last pushed commit on remote.

## Current JIT/dev workflow note

When changing `csrc/...` code in this environment, clear JIT artifacts before re-running:

```bash
cd /workspace/aiter-fresh
rm -rf aiter/jit/build
rm -f aiter/jit/module_*.so
```

## Suggested resume prompt

When resuming on another machine, use a prompt like:

`继续 split_k_mha_stage1_opus_handoff.md 的工作，从把 QK compute 替换成 opus::make_mfma 开始`

That should be enough context to continue without reconstructing the earlier design discussion.
