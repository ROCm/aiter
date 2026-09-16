# SILOTIGER-1047 notes

Harness pins, rocprof, and pasted tables live here from phase 1 onward.

## Phase 0

- Oracle: `aiter/ops/flydsl/kernels/qsa/oracle.py`
- Shapes: `aiter/ops/flydsl/kernels/qsa/shapes.py`
- Surface: `aiter/ops/flydsl/qsa.py` (no K1/K2 launcher yet)
- Gate: `HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py`

Tie-break: smaller block index on equal finite scores (`top_k_per_row_decode`).

Live AMD bar is vLLM **main** `qwen4_exp/amd/ops/qsa.py` (PR 53896 merged 2026-08-31). Record SHAs when the harness is wired.

## Phase 1a — family A plumbing

Paged caches: `aiter/ops/flydsl/kernels/qsa/paged.py` (`[n_pages, page_size, H, D]`
+ `block_table`). Indexer K is one compressed block per slot; GQA K/V are
uncompressed tokens. Physical pages are shuffled so a gather that ignores the
table cannot pass.

Sweep (defaults): `M ∈ {1, 8, 512}`, `L ∈ {512, 2048, 8192, 32768}`,
`page_size=16`, BF16. Oracle is not timed; `paged_gather` is the only
candidate. Pass `-s 131072` for 128k.

    HIP_VISIBLE_DEVICES=6 python3 -m pytest op_tests/test_flydsl_qsa.py -q
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py

## Phase 1b — live vLLM AMD path

Vendored from vllm-project/vllm **`836bb3839ffe`** (main, 2026-09-15),
file `vllm/models/qwen4_exp/amd/ops/qsa.py` (MQA + expand + sparse GQA only).
HIP top-k: `aiter.ops.topk._hip_top_k_per_row_decode` when
`aiter/jit/module_top_k_per_row.so` is present. At pin time this environment
could not JIT that module (`hipcub/hipcub.hpp` missing); the harness used the
oracle tie-break (`qsa_topk_blocks`) on vLLM MQA logits — same smaller-index
policy, **not** a substitute for the HIP kernel in production. Phase 1f
`vllm_amd_select` microseconds are that fallback column. The `.so` exists
later; do not read 1f select µs as HIP.

- Module: `aiter/ops/triton/_triton_kernels/attention/qsa_vllm_amd.py`
- Import: `aiter.ops.triton.attention.qsa_vllm_amd`
- aiter tree: `aa42c32a36bf` (this branch at pin time); `origin/main` was `797cce253bba`.

Table: `vllm_amd_select` (MQA+top-k+expand) and `vllm_amd_gqa` vs oracle
(block/token **set** equality; GQA `checkAllclose`). Oracle is not timed.

## Phase 1c — AITER #4882 Triton (no Gluon)

Vendored from ROCm/aiter **#4882** head **`150c7bc12b45`** (2026-09-15),
parent `2462d5b6427b`. Portable Triton only: `qsa_paged_mqa_logits`,
`qsa_expand_block_indices`, `qsa_sparse_paged_gqa` (`num_stages=2`). Gluon
kernels are not imported. HIP top-k: same as phase 1b at pin time (decode
`.so` or oracle tie-break), so `4882_triton_select` microseconds are that
same fallback, not HIP. Family A GQA is group 12 / D=256, so #4882 Gluon
would not auto-dispatch here anyway.

- Kernels: `aiter/ops/triton/_triton_kernels/attention/qsa_{paged_mqa_logits,expand_indices,sparse_paged_gqa}.py`
- Launchers: `aiter/ops/triton/_triton_kernels/attention/qsa_4882.py`
- Import: `aiter.ops.triton.attention.qsa_4882`

Table: `4882_triton_select` / `4882_triton_gqa` vs oracle (separate from the
vLLM AMD table). Do not merge family A vs B.

## Phase 1d — AITER #4882 Gluon (family B only)

Same pin **`150c7bc12b45`**. gfx950 Gluon kernels under
`aiter/ops/triton/_gluon_kernels/gfx950/attention/`. Launchers in
`qsa_4882.py` accept `backend="triton"|"gluon"|"auto"`; default is Triton so
family A stays a Triton column. Forced Gluon on family A GQA (group 12 / D=256)
errors. Family B table: `4882_gluon_select` / `4882_gluon_gqa` vs oracle,
indexer H ∈ {4, 8}. Skip on non-gfx950 or failed Gluon import.

