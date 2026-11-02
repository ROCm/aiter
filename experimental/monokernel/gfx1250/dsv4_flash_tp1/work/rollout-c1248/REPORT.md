# Flash C1/C2/C4/C8 rollout

Microbenchmark times are GPU microseconds on synthetic layer fixtures at position 1023, with the serving region bodies. Previous and updated monokernel paths are compared; these are not native ATOM unfused timings. Repeated paired graph timings, second round only.

| C | SWA previous | SWA updated | CSA previous | CSA updated | HCA previous | HCA updated |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 180.827 | 170.906 | 215.297 | 224.089 | 204.004 | 201.507 |
| 2 | 205.056 | 198.461 | 242.584 | 236.202 | 249.871 | 246.011 |
| 4 | 218.200 | 212.149 | 257.625 | 250.482 | 269.280 | 266.581 |
| 8 | 249.165 | 237.402 | 292.991 | 284.125 | 294.582 | 290.515 |

Each isolated group below is previous → updated, in microseconds. Unchanged groups are retained; no prefix-time subtraction.

| Group | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Attention mHC pre + input layout | 8.242 → 8.590 | 9.941 → 10.429 | 9.574 → 10.118 | 10.692 → 10.621 |
| CSA compressor/index | 13.173 → 13.220 | 13.705 → 13.694 | 13.523 → 13.528 | 13.965 → 13.957 |
| HCA compressor | 15.830 → 15.798 | 16.500 → 16.500 | 16.706 → 16.726 | 17.342 → 17.317 |
| Input quant + QKVA + norm/RoPE + QB | 18.876 → 18.881 | 20.451 → 20.409 | 19.477 → 19.519 | 21.258 → 21.250 |
| SWA attention + split-KV reduce | 8.441 → 8.469 | 8.760 → 8.755 | 8.517 → 8.530 | 9.420 → 9.398 |
| CSA attention + split-KV reduce | 9.454 → 9.513 | 9.900 → 9.914 | 9.753 → 9.773 | 10.517 → 10.484 |
| HCA attention + split-KV reduce | 8.517 → 8.511 | 9.195 → 9.201 | 9.080 → 9.080 | 9.622 → 9.621 |
| Inverse RoPE + quant + OA/OB | 17.646 → 17.611 | 19.328 → 19.283 | 21.177 → 21.877 | 25.811 → 26.370 |
| Attention mHC post + FFN mHC pre | 6.447 → 6.416 | 7.575 → 7.579 | 7.444 → 7.422 | 7.998 → 7.974 |
| Routing + shared/routed MoE + sum + final mHC post/layout | 45.733 → 42.671 | 50.469 → 48.340 | 67.848 → 65.414 | 87.705 → 84.613 |

End-to-end: Flash, one GPU per server, TP1, all 43 layers; 1024 input / 1024 output, C×8 requests. One server start per configuration; two client runs per concurrency; only round 2 shown. TTFT/TPOT are medians; total tokens/s includes input and output.

| C | Configuration | TTFT ms | TPOT ms | Total tok/s | Output tok/s |
|---:|---|---:|---:|---:|---:|
| 1 | native | 160.616 | 13.060 | 151.47 | 75.73 |
| 1 | previous | 160.820 | 9.993 | 196.97 | 98.48 |
| 1 | selected | 155.234 | 9.798 | 201.18 | 100.59 |
| 1 | replica | 168.458 | 9.837 | 200.27 | 100.13 |
| 2 | native | 298.669 | 13.322 | 294.10 | 147.05 |
| 2 | previous | 297.357 | 9.905 | 392.55 | 196.28 |
| 2 | selected | 292.886 | 9.663 | 403.74 | 201.87 |
| 2 | replica | 304.448 | 9.652 | 402.33 | 201.16 |
| 4 | native | 295.543 | 13.316 | 588.36 | 294.18 |
| 4 | previous | 284.691 | 10.900 | 715.95 | 357.97 |
| 4 | selected | 283.712 | 10.669 | 732.42 | 366.21 |
| 4 | replica | 301.233 | 10.588 | 735.78 | 367.89 |
| 8 | native | 290.431 | 13.627 | 1151.24 | 575.62 |
| 8 | previous | 290.033 | 11.917 | 1312.76 | 656.38 |
| 8 | selected | 282.579 | 11.638 | 1343.48 | 671.74 |
| 8 | replica | 290.144 | 11.620 | 1343.48 | 671.74 |

See [TTFT findings](TTFT_FINDINGS.md), [validation and limitations](VALIDATION.md), and [restore instructions](RESTORE.md). The selected configuration remains experimental; mixed-batch/native numerical parity is not qualified.
