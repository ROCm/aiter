# MXFP4 calibration results

Recorded manifest searches: 27. Valid winners: 27.

Regret is 100 × (candidate winner kernel-duration sum / reference winner kernel-duration sum − 1). Comparisons require matching valid finite winners. Full→prune measures M_est pruning; prune→staged measures stage screening. Missing full searches provide no full reference.

## Per-search evidence

Logged numerical errors below use the reference metric; CSV err1 is the separate rounded CSV field. Winner latency is the main-compatible kernel-duration sum. Tuning wall time and matched elapsed diagnostics are separate measurements.

| Search | Outcome | Valid winner | GEMM1 | GEMM2 | Winner sum (us) | Logged numerical errors | CSV err1 | Validations | Search wall (s) | Matched elapsed median (us) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimax-64-full | valid | True | flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16 | 76.9861 | 0.011493–0.011531 (1368 logged) | 1.1% | 1368 | 699.554425 | unavailable |
| minimax-64-prune | valid | True | flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16 | 75.8671 | 0.011494–0.011501 (192 logged) | 1.1% | 192 | 97.070872 | unavailable |
| minimax-64-staged | valid | True | flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16 | 76.0348 | 0.011435–0.011497 (10 logged) | 1.1% | 8 | 48.00212 | unavailable |
| minimax-128-prune | valid | True | flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16 | 84.5815 | 0.011322–0.011328 (1056 logged) | 1.1% | 1056 | 523.917856 | unavailable |
| minimax-128-staged | valid | True | flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16 | 84.2686 | 0.011304–0.011327 (20 logged) | 1.1% | 16 | 145.250049 | unavailable |
| minimax-512-prune | valid | True | flydsl_mxmoe_g1_a4w4_32x64x256_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32 | 97.7542 | 0.011286–0.011292 (1056 logged) | 1.1% | 1056 | 559.217337 | unavailable |
| minimax-512-staged | valid | True | flydsl_mxmoe_g1_a4w4_32x128x256_nt_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32 | 101.2191 | 0.011265–0.011292 (20 logged) | 1.1% | 16 | 158.590174 | unavailable |
| minimax-1024-prune | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_nt_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64 | 117.033 | 0.011207–0.011213 (1056 logged) | 1.1% | 1056 | 544.748742 | unavailable |
| minimax-1024-staged | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64 | 117.6557 | 0.011183–0.011213 (20 logged) | 1.1% | 16 | 157.420469 | unavailable |
| minimax-2048-full | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_nt_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_persist_nt_sbm64 | 173.7718 | 0.011173–0.01121 (1368 logged) | 1.1% | 1368 | 807.80887 | unavailable |
| minimax-2048-prune | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_nt_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64 | 174.7378 | 0.011173–0.011201 (456 logged) | 1.1% | 456 | 325.401227 | unavailable |
| minimax-2048-staged | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_persist_sbm64 | 179.6055 | 0.011153–0.01118 (30 logged) | 1.1% | 24 | 151.797219 | unavailable |
| minimax-4096-prune | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_persist_sbm64 | 248.1153 | 0.011185–0.011316 (360 logged) | 1.1% | 360 | 271.955927 | unavailable |
| minimax-4096-staged | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64 | 248.8316 | 0.01117–0.011212 (30 logged) | 1.1% | 24 | 137.728582 | unavailable |
| minimax-8192-full | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_scatter_sbm128 | 404.857 | 0.011168–0.011317 (1368 logged) | 1.1% | 1368 | 896.597213 | unavailable |
| minimax-8192-prune | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_scatter_sbm128 | 406.7968 | 0.011168–0.01134 (216 logged) | 1.1% | 216 | 185.393354 | unavailable |
| minimax-8192-staged | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_scatter_nt_sbm128 | 406.5586 | 0.011151–0.011183 (20 logged) | 1.1% | 16 | 110.136871 | unavailable |
| gpt-2048-default-prune | valid | True | flydsl_mxmoe_g1_a4w4_64x128x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_persist_nt_sbm64 | 115.6761 | 0.011468–0.011493 (912 logged) | 1.1% | 912 | 505.6521 | unavailable |
| gpt-2048-default-staged | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_swiglu_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_reduce_persist_sbm64 | 121.5918 | 0.01144–0.011483 (30 logged) | 1.1% | 24 | 209.593221 | unavailable |
| gpt-2048-fp8-prune | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128 | 101.864 | 0.011468–0.011838 (912 logged) | 1.2% | 912 | 493.998618 | unavailable |
| gpt-2048-fp8-staged | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_sbm128 | 110.5011 | 0.01144–0.011819 (30 logged) | 1.2% | 24 | 197.158416 | unavailable |
| gpt-2048-fp4-prune | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_nt_sbm128 | 111.0863 | 0.011468–0.018163 (912 logged) | 1.8% | 912 | 484.199742 | unavailable |
| gpt-2048-fp4-staged | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_nt_sbm128 | 110.3552 | 0.01144–0.01815 (30 logged) | 1.8% | 24 | 219.050345 | unavailable |
| gpt-2048-limit5-prune | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_swiglu_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_persist_nt_sbm64 | 117.0535 | 0.011468–0.011495 (912 logged) | 1.1% | 912 | 503.867533 | unavailable |
| gpt-2048-limit5-staged | valid | True | flydsl_mxmoe_g1_a4w4_64x256x256_swiglu_xcd4 | flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_reduce_persist_sbm64 | 122.0186 | 0.01144–0.011477 (30 logged) | 1.1% | 24 | 212.58035 | unavailable |
| gpt-2048-combined5-prune | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128 | 101.9686 | 0.01147–0.018171 (912 logged) | 1.2% | 912 | 476.701406 | unavailable |
| gpt-2048-combined5-staged | valid | True | flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2 | flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_sbm128 | 110.4887 | 0.01144–0.011819 (30 logged) | 1.2% | 24 | 203.993511 | unavailable |

## Paired outcomes and regret

| Model | Tokens | Configuration | Prune valid | Staged valid | Staged failure cause | Full→prune (%) | Prune→staged (%) | Full→staged (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimax | 64 | default | True | True | unavailable | -1.4535091399616284 | 0.22104443164430165 | -1.2356776093346622 |
| minimax | 128 | default | True | True | unavailable | unavailable | -0.3699390528661639 | unavailable |
| minimax | 512 | default | True | True | unavailable | unavailable | 3.544502435700969 | unavailable |
| minimax | 1024 | default | True | True | unavailable | unavailable | 0.5320721505899906 | unavailable |
| minimax | 2048 | default | True | True | unavailable | 0.5559014753832248 | 2.7857166566135128 | 3.357103971990849 |
| minimax | 4096 | default | True | True | unavailable | unavailable | 0.2886964246058277 | unavailable |
| minimax | 8192 | default | True | True | unavailable | 0.47913213801416 | -0.058555032881280056 | 0.4202965491519217 |
| gpt | 2048 | default | True | True | unavailable | unavailable | 5.114020960250221 | unavailable |
| gpt | 2048 | fp8 | True | True | unavailable | unavailable | 8.479050498704144 | unavailable |
| gpt | 2048 | fp4 | True | True | unavailable | unavailable | -0.658136961983613 | unavailable |
| gpt | 2048 | limit5 | True | True | unavailable | unavailable | 4.241735616619757 | unavailable |
| gpt | 2048 | combined5 | True | True | unavailable | unavailable | 8.355611433323595 | unavailable |

## Prune winner screening ranks

Ranks use successful staged screens, ascending score with original log order for ties. GEMM1 ranks are reused per BM; GEMM2 ranks separate atomic and non-atomic accumulation, combining reduce/scatter in the latter. Pair validated means a returned validation outcome was logged, including failure.

| Model | Tokens | Configuration | BM | Accumulation | GEMM1 rank | GEMM2 rank | Both retained K2 | Pair validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimax | 64 | default | 16 | atomic | 2 | 1 | True | True |
| minimax | 128 | default | 16 | atomic | 1 | 3 | False | False |
| minimax | 512 | default | 32 | atomic | 4 | 1 | False | False |
| minimax | 1024 | default | 64 | non_atomic | 2 | 1 | True | True |
| minimax | 2048 | default | 64 | non_atomic | 5 | 2 | False | False |
| minimax | 4096 | default | 64 | non_atomic | 2 | 4 | False | False |
| minimax | 8192 | default | 128 | non_atomic | 1 | 1 | True | True |
| gpt | 2048 | default | 64 | atomic | 2 | 3 | False | False |
| gpt | 2048 | fp8 | 128 | non_atomic | 2 | 7 | False | False |
| gpt | 2048 | fp4 | 128 | non_atomic | 2 | 2 | True | True |
| gpt | 2048 | limit5 | 64 | atomic | 2 | 3 | False | False |
| gpt | 2048 | combined5 | 128 | non_atomic | 1 | 7 | False | False |

Prune-valid/staged-invalid outcomes: 0.

| Failure cause | Count |
| --- | --- |

## Tuning wall-time totals

Observed sums include failed searches with recorded finite wall times. A partial sum does not estimate missing work; default MiniMax and extended GPT-OSS results remain separate.

| Matrix | Mode | Recorded / expected | Observed sum (s) | Complete |
| --- | --- | --- | --- | --- |
| default_minimax | prune | 7 / 7 | 2507.705315 | True |
| default_minimax | staged | 7 / 7 | 908.925484 | True |
| extended_gpt | prune | 5 / 5 | 2464.419399 | True |
| extended_gpt | staged | 5 / 5 | 1042.375843 | True |
