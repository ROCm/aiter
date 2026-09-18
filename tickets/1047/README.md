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

## Phase 2d — emit / ``visible <= 512`` vs live AMD select (HIP top-k)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Re-bench after
`aiter/jit/module_top_k_per_row.so` landed. Log shows
`import [module_top_k_per_row]`; oracle-fallback warning did not fire.
AMD column is Triton MQA + `_hip_top_k_per_row_decode` + expand. Oracle
set equality `err=0` on both columns (HIP `stable=False` still matched
this seed).

The FlyDSL column keeps the **`n_blocks <= 512` fast path**: write every
complete-block id and skip scoring. Longer rows use BLOCK_N=32 BF16 MFMA
scorer workgroups and a global fp32 `[M, n_blocks]` score buffer. Selection
is `flydsl_top_k_per_row_decode(stable=True)` below 32768 columns and
`topk_select(..., tie='low')` streaming radix at or above that width. H=4
is padded to 16 MFMA rows; D=128 is split across two waves. BLOCK_N=16
remains buildable but lost on prefill. Single-request prefill uses the MFMA M
dimension for 16 query rows and reuses each K tile across them; decode and
multi-request inputs keep the one-row scorer. Expand remains separate.

A 64-bit MSD binary radix-select on the 1024-candidate tile was measured
and not shipped. Set equality held. Decode ``M=1`` 8k / 32k ~126 / ~497 µs
vs kept bitonic ~106 / ~419 µs. Sixty-four digit passes vs 55 sort stages.

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 2.0 | 9.5 | 0 | 0 |
| 1 | 2048 | 512 | 2.0 | 10.3 | 0 | 0 |
| 1 | 8192 | 2048 | 10.5 | 19.1 | 0 | 0 |
| 8 | 8192 | 2048 | 12.7 | 22.0 | 0 | 0 |
| 1 | 32768 | 8192 | 14.1 | 20.3 | 0 | 0 |
| 8 | 32768 | 8192 | 19.4 | 25.2 | 0 | 0 |
| 1 | 131072 | 32768 | 28.3 | 30.0 | 0 | 0 |
| 8 | 131072 | 32768 | 46.3 | 52.9 | 0 | 0 |
| 512 | 512 | 128 | 3.5 | 18.1 | 0 | 0 |
| 512 | 2048 | 512 | 3.6 | 29.1 | 0 | 0 |
| 512 | 8192 | 2048 | 29.3 | 83.1 | 0 | 0 |
| 512 | 32768 | 8192 | 68.2 | 247.7 | 0 | 0 |

Streaming radix at 32768 columns moves 128k decode from 33.7 / 54.4 us to
28.3 / 46.3 us (`M=1` / `M=8`) and crosses live AMD. The 16-row scorer moves
8k / 32k prefill from 83.2 / 288.7 us to 29.3 / 68.2 us and beats live AMD
at both points.

## Phase 2e — family B K1 H=4 emit (not a win)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Kernel:
`aiter/ops/flydsl/kernels/qsa/k1_family_b.py`. Public:
`qsa_k1_family_b_block_ids` (``H=4`` in 2e). Oracle set equality `err=0`.
#4882 columns are Triton and Gluon select (HIP top-k on the competitor
path). Expand still separate. Separate table from family A. **Not a win
claim.** Long-`L` is 2g.

| m | seq_len | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 512 | 128 | 2.0 | 13.1 | 13.2 | 0 |
| 8 | 512 | 128 | 3.0 | 15.8 | 15.7 | 0 |
| 1 | 2048 | 512 | 2.0 | 10.2 | 10.4 | 0 |
| 8 | 2048 | 512 | 3.0 | 11.4 | 11.5 | 0 |

## Phase 2f — family B K1 H=8 emit (not a win)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Same kernel file; ``H=8``
is a second compile (``q.shape[1]``). Oracle set equality `err=0`. #4882
Triton and Gluon select. Expand still separate. **Not a win claim.**

| m | seq_len | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 512 | 128 | 2.0 | 13.8 | 13.2 | 0 |
| 8 | 512 | 128 | 2.8 | 16.2 | 15.5 | 0 |
| 1 | 2048 | 512 | 2.0 | 11.0 | 10.3 | 0 |
| 8 | 2048 | 512 | 2.9 | 12.1 | 11.4 | 0 |

## Phase 2g — family B K1 long-L tile merge (not a win)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Same kernel: 512-slot
tiles, running LDS top-512, no score matrix. Oracle set equality `err=0`
at 8k / 32k / 128k for ``H`` 4 and 8. #4882 Triton and Gluon select.
**Recorded loss**; do not chase a select win. 2h is emit / short-L only.

| m | seq_len | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|--:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 8192 | 4 | 2048 | 106.4 | 16.3 | 16.2 | 0 |
| 8 | 8192 | 4 | 2048 | 107.7 | 18.3 | 18.1 | 0 |
| 1 | 32768 | 4 | 8192 | 418.8 | 18.4 | 18.4 | 0 |
| 8 | 32768 | 4 | 8192 | 422.9 | 23.1 | 21.6 | 0 |
| 1 | 131072 | 4 | 32768 | 1716.1 | 27.7 | 27.2 | 0 |
| 8 | 131072 | 4 | 32768 | 1734.2 | 49.5 | 47.7 | 0 |
| 1 | 8192 | 8 | 2048 | 123.4 | 17.0 | 16.2 | 0 |
| 8 | 8192 | 8 | 2048 | 124.8 | 19.4 | 18.2 | 0 |
| 1 | 32768 | 8 | 8192 | 487.2 | 19.4 | 18.5 | 0 |
| 8 | 32768 | 8 | 8192 | 492.2 | 26.4 | 22.1 | 0 |
| 1 | 131072 | 8 | 32768 | 2022.0 | 29.9 | 27.4 | 0 |
| 8 | 131072 | 8 | 32768 | 2036.2 | 56.2 | 45.3 | 0 |

## Phase 2h — family B K1 emit vs #4882 (closed)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. **2h is closed** on emit /
``visible <= 512`` (``H`` 4 and 8, 2e/2f tables) plus the published
indexer point: ``M=32``, ``H=4``, ``D=128``, ``page_size=8``,
``n_blocks=512`` (512 compressed keys, 64 pages). Oracle set equality
`err=0`. Long-L is 2i (family A scorer), not a 2h gate.

| m | seq_len | page_size | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|----------:|--:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 32 | 2048 | 8 | 4 | 512 | 2.9 | 12.1 | 11.9 | 0 |

## Phase 2i — family B K1 long-L uses family A scorer

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Long rows dispatch into
family A's BLOCK_N=32 MFMA scorer plus decode/streaming radix. ``H=4``
shares that compile; ``H=8`` is a second compile. Emit kernels no longer
contain a bitonic path. Oracle set equality. 8k / 32k beat both #4882
columns; decode ``M=1`` 128k is a small loss vs Gluon.

| m | seq_len | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us |
|--:|--------:|--:|---------:|-------------:|----------------------:|---------------------:|
| 1 | 512 | 4 | 128 | 2.0 | 13.5 | 13.3 |
| 1 | 8192 | 4 | 2048 | 10.5 | 16.3 | 16.2 |
| 1 | 32768 | 4 | 8192 | 14.2 | 18.4 | 18.5 |
| 1 | 131072 | 4 | 32768 | 28.6 | 28.0 | 27.2 |
| 8 | 512 | 4 | 128 | 2.6 | 15.8 | 15.8 |
| 8 | 8192 | 4 | 2048 | 12.8 | 18.2 | 18.1 |
| 8 | 32768 | 4 | 8192 | 19.8 | 23.4 | 21.9 |
| 8 | 131072 | 4 | 32768 | 46.5 | 50.7 | 48.1 |
| 1 | 512 | 8 | 128 | 2.0 | 14.2 | 13.3 |
| 1 | 8192 | 8 | 2048 | 10.7 | 17.0 | 16.3 |
| 1 | 32768 | 8 | 8192 | 14.4 | 19.4 | 18.6 |
| 1 | 131072 | 8 | 32768 | 28.6 | 30.0 | 27.5 |
| 8 | 512 | 8 | 128 | 2.7 | 16.3 | 15.6 |
| 8 | 8192 | 8 | 2048 | 12.8 | 19.4 | 18.1 |
| 8 | 32768 | 8 | 8192 | 19.9 | 26.2 | 21.9 |
| 8 | 131072 | 8 | 32768 | 44.9 | 56.4 | 45.1 |

## Phase 3a — family A FlyDSL K2 decode (correctness)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Wrapper
`qsa_k2_family_a` writes ``o [M, 24, 256]`` from paged K/V at expanded
token ids. Oracle ``checkAllclose`` `err=0` (rtol/atol `1e-2`). No RoPE,
no sigmoid, expand still separate. Times vs live AMD GQA are **not** a
win claim.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 1 | 512 | 128 | 2051 | 10638.0 | 11.0 | 0 | 0 |
| 8 | 512 | 128 | 2051 | 10685.7 | 14.4 | 0 | 0 |
| 1 | 2048 | 512 | 2051 | 10685.4 | 11.2 | 0 | 0 |
| 8 | 2048 | 512 | 2051 | 10839.6 | 15.7 | 0 | 0 |
| 1 | 8192 | 2048 | 2051 | 10684.5 | 11.3 | 0 | 0 |
| 8 | 8192 | 2048 | 2051 | 10968.6 | 16.3 | 0 | 0 |
| 1 | 32768 | 8192 | 2051 | 10706.1 | 11.2 | 0 | 0 |
| 8 | 32768 | 8192 | 2051 | 11013.9 | 16.2 | 0 | 0 |

## Phase 3b — family A FlyDSL K2 split-K (decode occupancy)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Same `qsa_k2_family_a`
ABI: split-K along the expanded list (64 splits when ``M * Hk <= 8``)
plus an LSE merge. Oracle `err=0`. ~100× vs 3a at ``M=1`` (~103 µs vs
~10.6 ms); still ~10× live AMD (~10 µs). Not a win claim.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 1 | 512 | 128 | 2051 | 101.3 | 10.0 | 0 | 0 |
| 8 | 512 | 128 | 2051 | 236.8 | 13.3 | 0 | 0 |
| 1 | 2048 | 512 | 2051 | 103.3 | 10.2 | 0 | 0 |
| 8 | 2048 | 512 | 2051 | 247.9 | 16.0 | 0 | 0 |
| 1 | 8192 | 2048 | 2051 | 103.6 | 10.3 | 0 | 0 |
| 8 | 8192 | 2048 | 2051 | 245.7 | 16.2 | 0 | 0 |
| 1 | 32768 | 8192 | 2051 | 103.7 | 10.2 | 0 | 0 |
| 8 | 32768 | 8192 | 2051 | 245.0 | 16.2 | 0 | 0 |

## Phase 3c — family A FlyDSL K2 prefill (same instantiation)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Prefill ``M=512`` uses
the 3b split-K kernel (no second compile, no MFMA union). Oracle `err=0`.
Host uses 1 split (``M * Hk > 512``). ~11.7 ms vs live AMD ~201–285 µs.
Not a win claim.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 512 | 512 | 128 | 2051 | 11690.7 | 200.7 | 0 | 0 |
| 512 | 2048 | 512 | 2051 | 11709.1 | 211.8 | 0 | 0 |
| 512 | 8192 | 2048 | 2051 | 11831.8 | 246.8 | 0 | 0 |
| 512 | 32768 | 8192 | 2051 | 11884.8 | 284.6 | 0 | 0 |

## Phase 3d — family A FlyDSL K2 tiled MFMA QK/PV (not closed)

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Same ABI:
``BLOCK_N=16`` 128-bit paged gather, QK/PV MFMA (heads padded to 16),
split-K + LSE merge. gfx950 uses K32 QK; gfx942 retains K16. Softmax
runs on wave 0 off the reduced C fragment (no ``s`` LDS tile), and a
loop-top barrier separates the next gather from the previous tile's PV
reads. The next K/V tile is prefetched into registers before current QK
and carried across the runtime loop. Split outputs are BF16 and one merge
wave computes LSE weights once. Oracle `err=0` at decode and prefill.
Decode **beats #4882 Triton** (~18.7 µs vs ~113 µs at ``M=1``) but
**does not** beat live AMD (~10 µs). Prefill ~0.52–0.55 ms vs
~203–286 µs AMD / ~222–282 µs #4882. Not a win claim.

Decode:

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | 4882_triton_gqa us | flydsl_k2 err |
|--:|--------:|---------:|------:|-------------:|----------------:|-------------------:|--------------:|
| 1 | 512 | 128 | 2051 | 18.7 | 10.0 | 112.9 | 0 |
| 8 | 512 | 128 | 2051 | 23.0 | 13.3 | 115.5 | 0 |
| 1 | 2048 | 512 | 2051 | 19.1 | 10.3 | 115.9 | 0 |
| 8 | 2048 | 512 | 2051 | 24.5 | 16.0 | 126.3 | 0 |
| 1 | 8192 | 2048 | 2051 | 19.1 | 10.3 | 117.4 | 0 |
| 8 | 8192 | 2048 | 2051 | 24.7 | 15.1 | 121.9 | 0 |
| 1 | 32768 | 8192 | 2051 | 19.0 | 10.2 | 118.7 | 0 |
| 8 | 32768 | 8192 | 2051 | 25.0 | 15.4 | 123.1 | 0 |

Prefill:

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | 4882_triton_gqa us | flydsl_k2 err |
|--:|--------:|---------:|------:|-------------:|----------------:|-------------------:|--------------:|
| 512 | 512 | 128 | 2051 | 516.5 | 203.2 | 222.2 | 0 |
| 512 | 2048 | 512 | 2051 | 530.7 | 214.1 | 236.9 | 0 |
| 512 | 8192 | 2048 | 2051 | 546.5 | 248.3 | 260.3 | 0 |
| 512 | 32768 | 8192 | 2051 | 548.7 | 285.7 | 281.5 | 0 |

A gfx950 ``BLOCK_N=64`` second compile was measured and not shipped:
``splits=1`` ~1.01–1.17 ms, ``splits=8`` ~1.12–1.29 ms, both slower than
``BLOCK_N=16`` / 8-split (~0.54–0.57 ms). ~90 KB LDS; gfx942 cannot hold
that tile.

All-wave softmax plus ``shuffle_idx`` P transpose (drop the P LDS
barrier) was measured and not shipped: decode ~19.8–20.1 µs at ``M=1``,
prefill ~0.66 ms with sparse ``1e-2`` misses at ``M=512``.

128-bit LSE-merge loads along ``D`` were measured and not shipped:
decode ``M=1`` ~24.5 µs vs kept ~18.7–19.1 µs. Scalar per-``D`` merge
stays.

## Phase 3d — live-AMD-shaped FlyDSL replacement

The previous BLOCK_N=16/four-wave-only K2 implementation is removed.
The replacement builder mirrors live AMD's host policy:

- decode: BLOCK_N=16, 256 threads, target splits 64/32;
- larger rows: BLOCK_N=64, 128 threads, target splits 8/4/1;
- splits are capped to the largest useful power of two;
- split bounds divide complete BLOCK_N tiles, matching live AMD;
- one owner per column translates logical token to physical page/off;
- splits=1 writes output directly and does not launch merge;
- split output is FP32 and merge uses two waves.

The single-stage kernel aliases K and transposed-V LDS, so BLOCK_N=64
uses about 43 KiB rather than the old experiment's ~90 KiB. Q MFMA
fragments stay in registers across the tile loop. QK is K32 on gfx950
and K16 on gfx942; PV is K16. Online softmax and merge weights use
log2-space ``exp2``.

GPU 6 / gfx950 / ``FLYDSL_RUNTIME_ENABLE_CACHE=0``: the full QSA pytest
suite passes (18 tests), and width-2051 ``L=512`` correctness has
``err=0``:

| m | flydsl port µs | live AMD µs | #4882 Triton µs |
|--:|----------------:|------------:|-----------------:|
| 1 | 23.3 | 9.9 | 112.8 |
| 8 | 22.2 | 13.5 | 115.4 |
| 512 | 722.2 | 202.7 | 222.3 |

The port is structurally complete but does not meet the phase-3 gate.
The first authoring-alignment pass now stages both K and V through
``make_tiled_copy`` plus ``UniversalCopy128b`` into a shared row-major
``[BLOCK_N, D]`` layout, uses ``idx2crd`` for wave/lane coordinates, and
issues QK/PV through ``fx.gemm``. This removes the scalar V transpose and
turns the main K/V LDS stores into 128-bit writes; P/C traffic and the
per-lane MFMA feeds are still scalar.

GPU 6 / gfx950 / ``FLYDSL_RUNTIME_ENABLE_CACHE=0``: all 18 QSA pytest
cases pass. At width 2051 / ``L=512`` the aligned path is 23.8 / 24.1 /
588.1 us for ``M=1/8/512`` with ``err=0``. The prior port was 23.3 /
22.2 / 722.2 us, so prefill improves about 19% while decode is flat to
worse.

Occupancy ISA (``FLYDSL_DUMP_IR``, width 2051). The pytest decode case
is ``ns=1`` because width is 8; the bench kernel is ``BN=16`` / 256
threads / 32 splits for ``M=1``. Counts are unrolled static instructions, not
dynamic per-tile issue.

| kernel | ds_write b128/b32/b16 | ds_read b128/b64/b32/b16 | barrier | MFMA | VGPR | LDS B |
|---|---:|---:|---:|---:|---:|---:|
| decode split `bn16_blk256_ns32` | 5 / 15 / 4 | 4 / 1 / 12 / 0 | 6 | 6 | 79 | 21376 |
| decode merge `ns32_blk128` | 0 / 2 / 0 | 8 / 0 / 1 / 0 | 1 | 0 | 112 | 260 |
| prefill split `bn64_blk128_ns1` | 36 / 17 / 16 | 18 / 0 / 13 / 0 | 6 | 48 | 257 | 76736 |

There are **no** ``ds_read_b16``; QK/PV LDS reads already widen to
``b32``/``b64``/``b128``. Remaining ``ds_write_b16`` are P (and some C)
stores. Split kernels keep **6 barriers**. Prefill is store/MFMA-heavy
with a larger dual-KV LDS footprint; decode split is still small and
the 32-split merge does 8 ``ds_read_b128``.

QK B now loads through wave ``make_tiled_copy_B`` (128-bit on gfx950
K32, 64-bit on K16) into the MFMA B fragment instead of
``k_lds[n, d]`` scalars. GPU 6 / gfx950: 18 pytest cases, ``err=0``,
width-2051 ``L=512`` is 24.0 / 24.1 / 588.6 us — flat vs 23.8 / 24.1 /
588.1. ISA is unchanged on decode (VGPR 69 vs 70); prefill
``ds_read_b32`` went 11 → 13. The compiler already widened the old
scalar QK reads, so this copy is authoring-correct but not a wall-time
win.

**Do not retry** MMA-native PV A/B (`make_tiled_copy_A/B` or a TV copy
of P) while V stays row-major ``[BN, D]``. copy_B wants ``(D, tokens)``
with contiguous K; tokens are stride-D here (oracle ``err≈0.98``). A
matching 64-bit P TV copy was ISA-neutral, added decode VGPR, and was
reverted.

**Do not retry** 128-bit C stores plus shuffle-pack 64-bit P stores.
They matched the oracle but width-2051 ``L=512`` went 24.5 / 24.8 /
637.3 us vs 24.0 / 24.1 / 588.6.

The V-gather → softmax barrier is gone: softmax only reads C/live,
already published by the post-QK barrier, and V/P meet before PV.
Occupancy ISA is **6** split-kernel barriers (was 7). GPU 6 / gfx950:
18 pytest cases, ``err=0``, width-2051 ``L=512`` is 24.3 / 24.1 /
586.7 us — flat vs 24.0 / 24.1 / 588.6. Decode VGPR 69; prefill VGPR
165 vs 169. Do **not** retry all-wave softmax to drop the P barrier.

**Do not retry** per-thread redundant page translate to drop the
translate barrier. Oracle ``err=0``, but width-2051 ``L=512`` went
24.1 / 23.8 / 608.9 us vs 24.3 / 24.1 / 586.7.

**Do not retry** MMA-native QK A (`make_tiled_copy_A` on global Q).
Compile aborted in `CopyOpUniversalCopyType::emitAtomCallSSA`. Keep
the 128-bit Q `g_copy` plus `q_off` extract.

gfx950 keeps separate K and V ``[BN, D]`` LDS tiles (gfx942 still
aliases). Same tile order; the post-QK barrier still publishes C.
GPU 6 / gfx950: 18 pytest cases, ``err=0``, width-2051 ``L=512`` is
24.4 / 24.1 / 519.3 us vs 24.3 / 24.1 / 586.7 (~11% prefill). Decode
split LDS 21376 VGPR 79; prefill LDS 76736 VGPR 257; 6 barriers.

For ``M * Hk <= 4``, 32 decode splits replaces 64: width-2051
``L=512`` ``M=1`` improves from 24.4 to 20.7 us (~15%), ``M=8`` stays
on 32 splits and is flat at 24.0 us, and prefill stays 519.5 us. The
32-split merge has 8 ``ds_read_b128`` versus 34 at 64 splits.
**Do not retry** 16 splits for ``4 < M * Hk < 32``; ``M=8`` regressed
to 31.3 us from 24.1 us.

**Do not retry** extra prefill split-K when ``M * Hk > 512``. Four
splits kept ``err=0`` and 18 tests, but width-2051 ``L=512`` prefill
went 519.5 → 548.4 us. Keep the one-split direct ``out`` write.

**Do not retry** next-K/PV overlap. Retrying with local
``@flyc.jit`` dispatch, localized page-map LDS views, and explicit
``index``-to-``Int64`` conversion compiles and gives ``err=0``, but
width-2051 ``L=512`` is 24.2 / 24.1 / 530.0 us versus 24.4 / 24.1 /
519.3 us. All 18 tests pass; the ~2.1% prefill regression was reverted.

On gfx950, V gather for the current tile now sits after the K barrier
and before QK. GPU 6 / gfx950: 18 pytest cases, ``err=0``, width-2051
``L=512`` is 19.8 / 22.7 / 513.5 us vs 20.7 / 24.0 / 519.5. Occupancy
ISA is unchanged.

**Do not retry** token-major ``(D, BLOCK_N)`` V LDS with per-element
stores and ``make_tiled_copy_B`` PV **on prefill / both launch paths**.
``err=0``, decode 18.6 / 21.2 us, prefill 613.0 us vs 513.5. Decode-only
``BLOCK_N=16`` gfx950 is the mapping that landed later.

**Do not retry** ``BLOCK_N=64`` / 8 splits / 128 threads for
``4 < M * Hk < 32``. ``M=8`` went 22.7 → 33.5 us (``err=0``); ``M=1``
and prefill were flat. Keep BN16 / 32 splits / 256 threads there.

**Do not retry** 256-thread ``BLOCK_N=64`` / ``splits=1`` prefill on
dual-KV LDS. Prefill went 513.5 → 719.9 us (``err=0``). Keep 128
threads. Not the old aliased ~90 KiB 256-thread BN64 mapping.

**Do not retry** eliding gfx950 K32 ``q_off`` on the Q 128-bit load.
``err=0``; width-2051 ``L=512`` was 19.8 / 23.3 / 514.9 us vs 19.8 /
22.7 / 513.5. Prefill VGPR is not that extract.

**Do not retry** a second token-major ``v_t`` filled from the 128-bit
row-major V gather plus tiled PV B. Prefill went 513.5 → 935.7 us
(``err=0``). Keep one row-major V tile.

**Do not retry** 64-bit ``UniversalCopy`` PV-A loads from contiguous
``p_lds[h, n:n+4]``. Oracle ``err=0``; width-2051 ``L=512`` was 19.8 /
23.1 / 516.0 us vs 19.8 / 22.7 / 513.5. The 16 ``ds_read_u16`` P reads
are not the split-kernel gap. Keep scalar P gathers.

**Do not retry** ``ds_read_b64_tr_b16`` PV-B from unswizzled row-major
``[BLOCK_N, D]`` V (lane groups of 4, address ``(n0+lane_m%4)*D + (d-lane_m%4)``).
Decode oracle ``err≈0.98``. Same class of miss as MMA-native PV B on this
layout. Keep scalar ``v_lds[n, d]`` gathers.

**Do not retry** decode-only wave-0 full-D QK (8×K32 on wave 0, skip the
cross-wave C sum). 18 pytest cases, ``err=0``; width-2051 ``L=512`` was
20.5 / 24.0 / 513.0 us vs 19.8 / 22.7 / 513.5. Prefill is unchanged
(128 threads still split D). Idle waves plus extra Q fragments do not
close the split-kernel gap. Keep D-split QK across four waves.

**Do not retry** decode all-wave full-D QK (every wave issues 8×K32,
softmax reads only wave-0 C, no C sum). 18 pytest cases, ``err=0``;
width-2051 ``L=512`` was 21.0 / 25.7 / 515.4 us vs 19.8 / 22.7 / 513.5.
Redundant QK matches Triton's K32 count but is 4× the D-split flops and
loses decode. Keep D-split QK across four waves.

Decode-only gfx950 ``BLOCK_N=16`` now stores V as token-major
``(D, BN)`` and reads PV-B with ``make_tiled_copy_B``. Prefill stays
row-major ``[BN, D]`` (the both-path token-major mapping above still
must not be retried). GPU 6 / gfx950: 18 pytest cases, ``err=0``,
width-2051 ``L=512`` is 18.5 / 21.1 / 516.2 us vs 19.8 / 22.7 / 513.5
(~6–7% decode). Still ~1.8× live AMD on the split kernel.

**Do not retry** unrolling the 32-split LSE merge. ``err=0``; width-2051
``L=512`` was 18.6 / 21.2 / 513.0 us vs 18.5 / 21.1 / 516.2. Decode is
flat; prefill does not launch merge. Keep the runtime split loop.

**Do not retry** a 256-thread LSE merge (one ``D`` lane per thread).
``err=0``; width-2051 ``L=512`` was 18.6 / 21.3 / 513.6 us vs 18.5 /
21.1 / 516.2. Decode is slightly worse; prefill does not launch merge.
Keep two waves and two ``D`` lanes per thread.

**Do not retry** decode-only ``make_tiled_copy_A`` PV-A from
``p_lds[h, n]`` (64-bit atom, wave slice) with token-major V already
on copy_B. 18 pytest cases, ``err=0``; width-2051 ``L=512`` was 18.8 /
21.7 / 513.4 us vs 18.5 / 21.1 / 516.2. Same class of miss as 64-bit
scalar-pack P loads. Keep scalar ``p_lds[lane_m, n0:n0+4]`` gathers.

**Do not retry** shuffle-pack 128-bit token-major V LDS stores
(``shuffle_xor`` 8 tokens per D, ``UniversalCopy128b`` from
``col % 8 == 0``). 18 pytest cases, ``err=0``; width-2051 ``L=512`` was
24.7 / 29.1 / 516.5 us vs 18.5 / 21.1 / 516.2. Decode lost ~33%; shuffle
cost dominates scalar ``ds_write_b16``. Keep per-element
``v_lds[d, col]`` stores.














