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
`.so` or oracle tie-break). Phase 1f `4882_triton_select` microseconds are
that same fallback, not HIP. Family A GQA is group 12 / D=256, so #4882
Gluon would not auto-dispatch here anyway.

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

## Phase 1e — rocprof one live-AMD QSA layer (GPU 6)

Driver: `tickets/1047/profile_qsa_layer.py` (family A vLLM AMD select + sparse GQA;
oracle not run). Device: Instinct MI355X, `HIP_VISIBLE_DEVICES=6`, rocprofv3 1.3.2.
HIP `module_top_k_per_row.so` still missing; select top-k is the **oracle
`torch.topk` fallback**, so decode select wall time includes many ATen/rocprim
sort kernels, not production HIP radix top-k.

HIP graph: `torch.cuda.CUDAGraph` capture of the **full layer** succeeded at
decode `M=1` for `L ∈ {512, 8192, 32768, 131072}` (internal logits allocs are
graph-pool safe). Replay is ~3× faster than eager layer (launch coalescing).

Event times (eager, not under rocprof; `--warmup/--iters` as in the driver):

| M | L | n_blocks | select_us | gqa_us | layer_us | graph_us |
|--:|--:|---------:|----------:|-------:|---------:|---------:|
| 1 | 512 | 128 | 148 | 37 | 192 | 56 |
| 1 | 8192 | 2048 | 132 | 37 | 178 | 65 |
| 1 | 32768 | 8192 | 143 | 37 | 188 | 76 |
| 1 | 131072 | 32768 | 157 | 39 | 203 | 88 |
| 8 | 8192 | 2048 | 126 | 37 | 167 | — |
| 8 | 32768 | 8192 | 188 | 39 | 235 | — |
| 512 | 8192 | 2048 | 157 | 266 | 407 | — |
| 512 | 32768 | 8192 | 524 | 290 | 801 | — |

rocprofv3 `--kernel-trace --stats` (includes warmup + eager + graph; named QSA
kernels only, mean µs):

| L | `_qsa_mqa_paged` | `_expand_qsa_indices` | `_qsa_sparse_paged_gqa_splitk` | `_qsa_merge_splitk` |
|--:|-----------------:|----------------------:|-------------------------------:|--------------------:|
| 512 | 3.31 | 2.95 | 6.29 | 3.60 |
| 32768 | 3.68 | 3.13 | 6.56 | 3.60 |

Raw CSVs were left in `/tmp/qsa_rocprof_{short,long}` (not in git).

**Indexer vs GQA (this GPU, live AMD path, fallback top-k):** do **not** swap
phases 2 vs 3 was the call at 1e. Decode select **wall** dominated GQA, but
that wall was fallback top-k + copies, not MQA (~3 µs) and not HIP radix.
That is **not** a ranking against production HIP select. “K1 must absorb the
expensive decode top-k” described the missing `.so`, not
`_hip_top_k_per_row_decode`. Prefill `M=512` under the same fallback: GQA
slightly ahead at 8k; select ahead at 32k. K2 still matters at prefill 8k.

