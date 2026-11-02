# Flash FFN: 1K input / 1K output, concurrency 64

Single GPU per experiment; independent experiments used all four gfx1250 GPUs. True M=64 inputs, hidden=4096, shared intermediate=2048, 256 routed experts, top-6 (HERD top-7 followed by pruning). These FFN dimensions do not depend on decode position. This is Flash TP1, not Pro TP4.

All times below are second-round GPU-event medians in microseconds. Both rounds and samples are retained in each component directory. Native implementations follow the installed ATOM methods and original serving defaults. Server was started once and clients run twice; client logs and the C64 server graph-capture trace are included. These are isolated operator benchmarks, not end-to-end serving speedups.

| ATOM operations | Candidate ablation | Native us | Candidate us |
|---|---|---:|---:|
| Attention mHC post + FFN mHC pre + RMSNorm + shared-input FP8 quant | Existing post/pre reduction with fused quant | 8.98222 | 5.96786 |
| Router BF16 GEMM (layer 0) | No custom M64 implementation | 3.67644 | — |
| HERD routing: population reset, top-7, prune/sort to 6, routing metadata | No alternative implementation | 9.81504 | — |
| Shared input FP8 quant | Native retained | 1.79848 | — |
| Shared gate/up GEMM | AITER Gluon WMMA/TDM ablation | 4.91955 | 4.90813 |
| Clamped SiLU times up | Native retained | 1.51336 | — |
| Shared down-input FP8 quant | Native retained | 1.77421 | — |
| Shared down GEMM | AITER Gluon WMMA/TDM ablation | 3.40354 | 3.26974 |
| Complete shared-expert chain (all five launches) | Same chain with candidate GEMMs | 16.47591 | 18.35982 |
| Routed input MXFP8 quant + gate/up/SwiGLU/requant + down + expert combine | Quant + persistent fused MoE (two launches) | 144.96882 | 362.83584 |
| Routed/shared BF16 add + extracted mHC post | New one-launch TDM combine_post | 5.19560 | 2.30141 |

A dash means no independent custom M64 candidate, not zero time. Rows overlap: do not sum the table. In particular shared-input quant is included in both the first row and shared chain, and shared GEMM rows are parts of that chain. Chained timings must be measured directly, not inferred from isolated sums.

## Boundaries and decisions

ATOM normally combines attention post with FFN pre. The previous full-attention monokernel already includes attention post: using the first row requires moving that boundary back before post. If attention post remains completed, native FFN-pre/RMSNorm alone is 8.70838 us; there is no standalone M64 candidate for that cut. Native post/pre/RMSNorm without quant is 5.89666 us.

The last row is an extracted endpoint: native add + residual unshuffle + HIP mHC post (three launches). Serving normally folds FFN post into the next attention pre. It is not a claim that serving launches these three kernels separately. Standalone BF16 add costs 1.27551 us.

Keep native shared-expert and routed-MoE paths at C64. The persistent MoE prototype is 2.50x slower; its 256 resident CTAs loop through tasks and use two grid barriers, whereas native expert GEMMs each launch 4224 CTAs. This is a design distinction, not a measured causal breakdown. The useful measured fusion opportunities are quant in the mHC reduction and add/post glue. No new complete C64 layer monokernel or serving integration is claimed.

The shared candidate uses existing Gluon BM32/BN32/BK256, four warps, three buffers, split-K 1. Its configuration was selected earlier at M8 and explicitly tested here at M64. There is no existing automatic M64 shared-expert dispatch entry. Both chains have five launches; the candidate is not a newly fused shared-expert kernel.

## Correctness and provenance

mHC outputs are bitwise equal across 16 M64 batches. Shared up/down/chain match bitwise across eight seeds. Routed MoE matches bitwise on the initial test and 16 further batches at fixed checkpoint-derived HERD routing. New combine_post matches bitwise across 16 batches. These are synthetic-input tests with checkpoint weights, not captured model activations, arbitrary routing validation, or GSM8K.

ATOM f19778df4054e94d64d6f6bf6dd68f5681bd5945; AITER 7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe. Container vigokhal_dsv4_tp4, original image rocm/fw-bringup-mlperf:gfx1250-atom-20261015. Actual C64 server capture confirms HERD routing; the current Triton MoE branch bypasses the nominal early-layer hash callback.

Exact kernel symbols, configs, profiles, assembly, validation, and reproduction commands are in mhc/REPORT.md, shared-router/REPORT.md, moe/README.md and combine-post/. The new combine_post uses 2048 CTAs, 30 VGPR, 1584 bytes LDS, no spills, no cross-CTA synchronization.

## Reproduction dependencies

Extract the archive into the directory mounted as /work in the compatible container. It includes ffn-c64 plus the graph helper and prior hybrid wrapper at their expected relative paths. ATOM, AITER, PyTorch/Triton from the experiment environment are required. The checkpoint /data/DeepSeek-V4-Flash remains an external dependency; the archive does not duplicate the model. Each region has bench.py; run with HIP_VISIBLE_DEVICES selecting an idle GPU and PYTHONPATH=/app/aiter:/app/ATOM. Both rounds are run by the harness. No serving process is required for isolated operator benchmarks.
