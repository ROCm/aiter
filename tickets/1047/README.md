# SILOTIGER-1047 notes

Harness pins, rocprof, and pasted tables live here from phase 1 onward.

## Phase 0

- Oracle: `aiter/ops/flydsl/kernels/qsa/oracle.py`
- Shapes: `aiter/ops/flydsl/kernels/qsa/shapes.py`
- Surface: `aiter/ops/flydsl/qsa.py`
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
errors. Family B Gluon table: `4882_gluon_select` / `4882_gluon_gqa` vs oracle,
indexer H ∈ {4, 8}. Skip on non-gfx950 or failed Gluon import. Family B
**Triton** is a separate table (phase 1f).

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

## Phase 1f — family A vs family B tables (never merged)

GPU 6 / gfx950 / MI355X. Sweep: `M ∈ {1, 8}`, `L ∈ {512, 8192, 32768}`,
`page_size=16`, BF16. All `err` columns were 0 vs the oracle. Prefill `M=512`
and `L=2048` were not in this paste; re-run the script defaults for those.

Family A: live AMD + #4882 Triton (Gluon does not dispatch GQA). Family B:
#4882 Triton and #4882 Gluon (no live-AMD column). Plumbing is family A only.

### Family A plumbing

|   m |   seq_len |   page_size | dtype          | gfx    |   n_blocks |   index_width |   paged_gather us |   paged_gather TFLOPS |   paged_gather TB/s |   paged_gather err |
|----:|----------:|------------:|:---------------|:-------|-----------:|--------------:|------------------:|----------------------:|--------------------:|-------------------:|
|   1 |       512 |          16 | torch.bfloat16 | gfx950 |        128 |          2051 |           48.113  |                     0 |           0.0224751 |                  0 |
|   1 |      8192 |          16 | torch.bfloat16 | gfx950 |       2048 |          2051 |           94.5615 |                     0 |           0.182966  |                  0 |
|   1 |     32768 |          16 | torch.bfloat16 | gfx950 |       8192 |          2051 |          189.415  |                     0 |           0.365368  |                  0 |
|   8 |       512 |          16 | torch.bfloat16 | gfx950 |        128 |          2051 |           48.4092 |                     0 |           0.0223376 |                  0 |
|   8 |      8192 |          16 | torch.bfloat16 | gfx950 |       2048 |          2051 |           94.8563 |                     0 |           0.182397  |                  0 |
|   8 |     32768 |          16 | torch.bfloat16 | gfx950 |       8192 |          2051 |          189.347  |                     0 |           0.365498  |                  0 |

### Family A vLLM AMD

|   m |   seq_len |   page_size | dtype          | gfx    | vllm_pin     |   n_blocks |   vllm_amd_select us |   vllm_amd_select TFLOPS |   vllm_amd_select TB/s |   vllm_amd_select err |   vllm_amd_gqa us |   vllm_amd_gqa TFLOPS |   vllm_amd_gqa TB/s |   vllm_amd_gqa err |
|----:|----------:|------------:|:---------------|:-------|:-------------|-----------:|---------------------:|-------------------------:|-----------------------:|----------------------:|------------------:|----------------------:|--------------------:|-------------------:|
|   1 |       512 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |        128 |              50.8827 |               0.00257597 |            0.000664116 |                     0 |           10.9761 |               4.59231 |           0.0977721 |                  0 |
|   1 |      8192 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |       2048 |              62.4135 |               0.0336009  |            0.00841664  |                     0 |           11.2938 |               4.46309 |           1.4877    |                  0 |
|   1 |     32768 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |       8192 |              79.0028 |               0.106181   |            0.0265582   |                     0 |           11.0797 |               4.54935 |           6.05915   |                  0 |
|   8 |       512 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |        128 |              62.0036 |               0.0169115  |            0.000660606 |                     0 |           14.2894 |              28.2197  |           0.0871404 |                  0 |
|   8 |      8192 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |       2048 |              73.4565 |               0.228397   |            0.00724892  |                     0 |           16.3966 |              24.5931  |           1.0352    |                  0 |
|   8 |     32768 |          16 | torch.bfloat16 | gfx950 | 836bb3839ffe |       8192 |             153.06   |               0.438447   |            0.013755    |                     0 |           15.8577 |              25.4288  |           4.24433   |                  0 |

### Family A #4882 Triton

|   m |   seq_len |   page_size | dtype          | gfx    | aiter_4882_pin                           |   n_blocks |   4882_triton_select us |   4882_triton_select TFLOPS |   4882_triton_select TB/s |   4882_triton_select err |   4882_triton_gqa us |   4882_triton_gqa TFLOPS |   4882_triton_gqa TB/s |   4882_triton_gqa err |
|----:|----------:|------------:|:---------------|:-------|:-----------------------------------------|-----------:|------------------------:|----------------------------:|--------------------------:|-------------------------:|---------------------:|-------------------------:|-----------------------:|----------------------:|
|   1 |       512 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 59.1434 |                  0.00221617 |               0.000571357 |                        0 |              112.964 |                 0.446206 |             0.00949991 |                     0 |
|   1 |      8192 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 63.0328 |                  0.0332708  |               0.00833394  |                        0 |              118.899 |                 0.423934 |             0.141311   |                     0 |
|   1 |     32768 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                 78.6155 |                  0.106704   |               0.0266891   |                        0 |              118.914 |                 0.423879 |             0.564552   |                     0 |
|   8 |       512 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 71.6257 |                  0.0146397  |               0.000571862 |                        0 |              114.999 |                 3.50648  |             0.0108278  |                     0 |
|   8 |      8192 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 73.3887 |                  0.228607   |               0.00725561  |                        0 |              125.665 |                 3.20888  |             0.135072   |                     0 |
|   8 |     32768 |          16 | torch.bfloat16 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                152.781  |                  0.439249   |               0.0137802   |                        0 |              127.384 |                 3.16558  |             0.528368   |                     0 |

### Family B #4882 Triton

|   m |   seq_len |   page_size | dtype          |   index_heads | gfx    | aiter_4882_pin                           |   n_blocks |   4882_triton_select us |   4882_triton_select TFLOPS |   4882_triton_select TB/s |   4882_triton_select err |   4882_triton_gqa us |   4882_triton_gqa TFLOPS |   4882_triton_gqa TB/s |   4882_triton_gqa err |
|----:|----------:|------------:|:---------------|--------------:|:-------|:-----------------------------------------|-----------:|------------------------:|----------------------------:|--------------------------:|-------------------------:|---------------------:|-------------------------:|-----------------------:|----------------------:|
|   1 |       512 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 60.6191 |                  0.00216222 |               0.000557448 |                        0 |              88.5576 |                 0.11858  |             0.00597812 |                     0 |
|   1 |       512 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 60.7462 |                  0.0043154  |               0.000573138 |                        0 |              88.5548 |                 0.118583 |             0.00597831 |                     0 |
|   1 |      8192 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 62.1363 |                  0.0337508  |               0.00845419  |                        0 |              92.783  |                 0.113179 |             0.0904662  |                     0 |
|   1 |      8192 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 63.2424 |                  0.0663211  |               0.00832252  |                        0 |              92.4382 |                 0.113602 |             0.0908037  |                     0 |
|   1 |     32768 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                 77.4369 |                  0.108328   |               0.0270953   |                        0 |              94.1222 |                 0.111569 |             0.356553   |                     0 |
|   1 |     32768 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                 78.8529 |                  0.212766   |               0.0266217   |                        0 |              94.2358 |                 0.111434 |             0.356123   |                     0 |
|   8 |       512 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 71.8155 |                  0.014601   |               0.00057035  |                        0 |              88.8031 |                 0.946014 |             0.00636518 |                     0 |
|   8 |       512 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                 72.6163 |                  0.0288799  |               0.000676872 |                        0 |              88.8235 |                 0.945797 |             0.00636372 |                     0 |
|   8 |      8192 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 73.2568 |                  0.229019   |               0.00726867  |                        0 |              94.3934 |                 0.889987 |             0.0893025  |                     0 |
|   8 |      8192 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                 74.3066 |                  0.451567   |               0.00727623  |                        0 |              94.3178 |                 0.890701 |             0.0893741  |                     0 |
|   8 |     32768 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                153.279  |                  0.437821   |               0.0137354   |                        0 |              94.9122 |                 0.885123 |             0.353963   |                     0 |
|   8 |     32768 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                155.385  |                  0.863774   |               0.0136019   |                        0 |              94.7417 |                 0.886716 |             0.3546     |                     0 |

### Family B #4882 Gluon

|   m |   seq_len |   page_size | dtype          |   index_heads | gfx    | aiter_4882_pin                           |   n_blocks |   4882_gluon_select us |   4882_gluon_select TFLOPS |   4882_gluon_select TB/s |   4882_gluon_select err |   4882_gluon_gqa us |   4882_gluon_gqa TFLOPS |   4882_gluon_gqa TB/s |   4882_gluon_gqa err |
|----:|----------:|------------:|:---------------|--------------:|:-------|:-----------------------------------------|-----------:|-----------------------:|---------------------------:|-------------------------:|------------------------:|--------------------:|------------------------:|----------------------:|---------------------:|
|   1 |       512 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                58.9352 |                 0.002224   |              0.000573376 |                       0 |             75.3928 |                0.139285 |            0.007022   |                    0 |
|   1 |       512 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                58.8476 |                 0.00445463 |              0.00059163  |                       0 |             75.3687 |                0.13933  |            0.00702424 |                    0 |
|   1 |      8192 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                62.4264 |                 0.033594   |              0.00841491  |                       0 |             81.9377 |                0.12816  |            0.10244    |                    0 |
|   1 |      8192 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                62.8791 |                 0.0667043  |              0.00837061  |                       0 |             81.5724 |                0.128734 |            0.102899   |                    0 |
|   1 |     32768 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                77.6398 |                 0.108045   |              0.0270245   |                       0 |             82.8506 |                0.126748 |            0.405061   |                    0 |
|   1 |     32768 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |                78.3119 |                 0.214236   |              0.0268056   |                       0 |             83.0514 |                0.126441 |            0.404081   |                    0 |
|   8 |       512 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                71.0751 |                 0.0147531  |              0.000576292 |                       0 |             75.6902 |                1.10991  |            0.00746792 |                    0 |
|   8 |       512 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |        128 |                72.0242 |                 0.0291173  |              0.000682437 |                       0 |             75.6503 |                1.11049  |            0.00747186 |                    0 |
|   8 |      8192 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                72.8378 |                 0.230337   |              0.00731049  |                       0 |             83.4849 |                1.00628  |            0.100971   |                    0 |
|   8 |      8192 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       2048 |                73.2527 |                 0.458064   |              0.00738092  |                       0 |             82.7189 |                1.0156   |            0.101906   |                    0 |
|   8 |     32768 |          16 | torch.bfloat16 |             4 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |               152.23   |                 0.440837   |              0.01383     |                       0 |             84.9735 |                0.988649 |            0.395363   |                    0 |
|   8 |     32768 |          16 | torch.bfloat16 |             8 | gfx950 | 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba |       8192 |               151.431  |                 0.886332   |              0.0139571   |                       0 |             84.2029 |                0.997696 |            0.398981   |                    0 |

## Phase 2a — family A FlyDSL K1 (correctness)

Kernel: `aiter/ops/flydsl/kernels/qsa/k1_family_a.py`. Public:
`qsa_k1_family_a_block_ids`. Bound: 512 page-aligned slots (`L <= 2048` at
`r=4`). No `[M, n_blocks]` score buffer. Expand still separate.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality. A static
TV-layout tiled copy stages Q in LDS, BF16 K rows use 128-bit buffer-copy
fragments, and a parallel bitonic merge keeps the best 512 of 1024 candidates.
**Not a win claim** vs live AMD. This env still lacks `module_top_k_per_row.so`.
These AMD microseconds are **not** the 2d bar.

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 97.7 | 43.5 | 0 | 0 |
| 8 | 512 | 128 | 102.0 | 54.5 | 0 | 0 |
| 1 | 2048 | 512 | 117.1 | 51.0 | 0 | 0 |
| 8 | 2048 | 512 | 121.5 | 67.4 | 0 | 0 |

## Phase 2b — family A FlyDSL K1 long-L merge

Same kernel streams 512-slot tiles into a running LDS top-512. No global
score matrix; no `topk_per_row_*` call. Expand still separate.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality. Each
tile is merged with the running top-512 by a 55-stage in-LDS bitonic network.
**Not a win claim.** This env still lacks `module_top_k_per_row.so`.
These AMD microseconds are **not** the 2d bar.

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 97.7 | 43.5 | 0 | 0 |
| 8 | 512 | 128 | 102.0 | 54.5 | 0 | 0 |
| 1 | 2048 | 512 | 117.1 | 51.0 | 0 | 0 |
| 8 | 2048 | 512 | 121.5 | 67.4 | 0 | 0 |
| 1 | 8192 | 2048 | 459.8 | 56.0 | 0 | 0 |
| 8 | 8192 | 2048 | 479.6 | 63.5 | 0 | 0 |
| 1 | 32768 | 8192 | 1844.8 | 69.4 | 0 | 0 |
| 8 | 32768 | 8192 | 1889.8 | 152.4 | 0 | 0 |
| 1 | 131072 | 32768 | 7415.0 | 96.2 | 0 | 0 |
| 8 | 131072 | 32768 | 7651.1 | 278.7 | 0 | 0 |

## Phase 2c — family A FlyDSL K1 prefill

Same wave64-per-row kernel as decode. Occupancy did not die at `M=512`, so
there is no second compile. Expand still separate.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality.
**Not a win claim.** This env still lacks `module_top_k_per_row.so`.
These AMD microseconds are **not** the 2d bar.

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 512 | 512 | 128 | 97.2 | 72.0 | 0 | 0 |
| 512 | 2048 | 512 | 117.3 | 92.0 | 0 | 0 |
| 512 | 8192 | 2048 | 458.4 | 161.1 | 0 | 0 |
| 512 | 32768 | 8192 | 1826.2 | 509.2 | 0 | 0 |

## Phase 2d — beat live AMD select (HIP top-k; not a win)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Re-bench after
`aiter/jit/module_top_k_per_row.so` landed. Log shows
`import [module_top_k_per_row]`; oracle-fallback warning did not fire.
AMD column is Triton MQA + `_hip_top_k_per_row_decode` + expand. Oracle
set equality `err=0` on both columns (HIP `stable=False` still matched
this seed).

The FlyDSL column is eight-wave K1 plus a **`visible <= 512` emit path**,
a **wave-local tile sort**, a **per-tile pair merge**, a **column split**
on decode rows with more than one tile (`S=8` while `n_tiles<=8`, else
`S=16`), **pipelined vec8 `score_col` that loads each MQA K column once**
across four Q heads, and a **4×512×128 MFMA** scorer on serial (16×16×16
BF16, 512×16 K panels in LDS, ReLU-sum). Prefill stays single-WG.
Workspace is `[M, S, 512]`, not a score matrix.
**2d stays unchecked:** short `L` beats HIP; 8k stays S=8 at ~22.3µs vs
HIP ~18; merge-kernel pair-merge shuffles strides 32..1. S=16 plus that
tail: decode 128k ~88µs vs HIP ~29; 32k ~48µs vs HIP ~19. S=32 lost at
128k (~105µs). Serial MFMA keeps prefill 8k/32k ~89/351µs vs HIP
~83/244. Keep `m > 8` serial.

rocprofv3 1.3.2 / GPU 6 (`tickets/1047/profile_qsa_k1.py`). Mean kernel µs
(35 launches), traced under **S=8**. Raw CSV in
`/tmp/qsa_k1_rocprof_{8k,32k}` (not in git). Split dropped
~16.9/34.1µs→~13.6/27.6µs; merge is unchanged. After dynamic S the gap
at 32k/128k is **merge-tree + remaining sequential tiles**, not more S:

| L | flydsl split | flydsl merge | HIP MQA | HIP radix top-k | HIP expand |
|--:|-------------:|-------------:|--------:|----------------:|-----------:|
| 8192 | 13.6 | 9.9 | 3.3 | 8.2 | 2.4 |
| 32768 | 27.6 | 19.0 | 3.3 | 10.2 | 2.3 |

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 1.5 | 7.2 | 0 | 0 |
| 8 | 512 | 128 | 2.4 | 8.9 | 0 | 0 |
| 1 | 2048 | 512 | 1.4 | 8.0 | 0 | 0 |
| 8 | 2048 | 512 | 2.4 | 9.1 | 0 | 0 |
| 1 | 8192 | 2048 | 22.3 | 18.2 | 0 | 0 |
| 8 | 8192 | 2048 | 24.0 | 19.9 | 0 | 0 |
| 1 | 32768 | 8192 | 47.8 | 19.4 | 0 | 0 |
| 8 | 32768 | 8192 | 48.6 | 24.0 | 0 | 0 |
| 1 | 131072 | 32768 | 88.4 | 29.1 | 0 | 0 |
| 8 | 131072 | 32768 | 97.2 | 52.2 | 0 | 0 |
| 512 | 512 | 128 | 2.7 | 16.0 | 0 | 0 |
| 512 | 2048 | 512 | 2.5 | 27.1 | 0 | 0 |
| 512 | 8192 | 2048 | 89.2 | 82.8 | 0 | 0 |
| 512 | 32768 | 8192 | 350.5 | 243.9 | 0 | 0 |






