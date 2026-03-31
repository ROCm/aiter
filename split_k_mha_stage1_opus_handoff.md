# split_k_mha stage1 opus handoff

## Scope

This note captures the current design direction for the split-K MHA stage1 kernel written in an `opus` style, so work can continue on another machine without reconstructing the context.

This is intentionally focused on **stage1 compute design**, not on wiring it into pybind/JIT yet.

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

## Why the skeleton file exists

`csrc/kernels/mha/prefill_splitk_stage1_opus_skeleton.cu` is **not** wired into build/JIT yet.

It exists to freeze:

- the launch contract
- the `work_info -> block` mapping
- the contiguous `Q/K/V` assumption
- the planned `opus` integration points

This avoids losing design progress while the actual implementation is still changing.

## What is already in the skeleton

The skeleton file already contains:

- `WorkInfoView`
- contiguous `Q/K/V` view helpers
- the simplified kernel signature
- `grid.x / grid.y` work/head mapping
- `opus::make_gmem` and `opus::make_smem` usage for loader structure
- shared-memory staging for `K/V`
- a reference-style scalar `QK / softmax / PV` loop
- TODO comments showing exactly where `make_mfma` should replace the scalar loops

## Agreed next compute step

The next implementation step is **not** more metadata work.

The next step is:

1. keep the skeleton launch / loader structure
2. replace `QK` scalar accumulation with `opus::make_mfma`
3. then replace `PV` scalar accumulation with `opus::make_mfma`

Recommended MFMA choice for first pass:

- `16x16x16`

Recommended decomposition for one wave:

- `Q stripe`: `[32, 128]`
- `K tile`: `[32, 128]`
- `Score`: `[32, 32]`
- decompose into `2 x 2 x 8` MFMA micro-tiles

For `PV`:

- `P tile`: `[32, 32]`
- `V tile`: `[32, 128]`
- `O tile`: `[32, 128]`
- use a logical transpose view of `V` for MFMA-B

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
