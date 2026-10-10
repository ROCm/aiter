# Current shared-enumeration counts

ROCm/FlyDSL imports with synthetic discovery; whole-process empty visibility; guarded GPU/JIT boundaries

Static admitted pairs, not runtime compilation or numerical validity

G1 ranks are reused across each BM's two accumulation groups. Reduce/scatter share the non-atomic group. K2 counts are initial shortlist upper counts; reference/score failures can exhaust all admitted pairs.

| Search | M_est | G1 | G2 | Pairs | K2 shortlist upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimax-64-full | 2 | 84 | 68 | 1368 | 32 |
| minimax-64-prune | 2 | 12 | 16 | 192 | 8 |
| minimax-64-staged | 2 | 12 | 16 | 192 | 8 |
| minimax-128-prune | 4 | 66 | 32 | 1056 | 16 |
| minimax-128-staged | 4 | 66 | 32 | 1056 | 16 |
| minimax-512-prune | 16 | 66 | 32 | 1056 | 16 |
| minimax-512-staged | 16 | 66 | 32 | 1056 | 16 |
| minimax-1024-prune | 32 | 66 | 32 | 1056 | 16 |
| minimax-1024-staged | 32 | 66 | 32 | 1056 | 16 |
| minimax-2048-full | 64 | 84 | 68 | 1368 | 32 |
| minimax-2048-prune | 64 | 27 | 52 | 456 | 24 |
| minimax-2048-staged | 64 | 27 | 52 | 456 | 24 |
| minimax-4096-prune | 128 | 21 | 52 | 360 | 24 |
| minimax-4096-staged | 128 | 21 | 52 | 360 | 24 |
| minimax-8192-full | 256 | 84 | 68 | 1368 | 32 |
| minimax-8192-prune | 256 | 12 | 36 | 216 | 16 |
| minimax-8192-staged | 256 | 12 | 36 | 216 | 16 |
| gpt-2048-default-prune | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-default-staged | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-fp8-prune | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-fp8-staged | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-fp4-prune | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-fp4-staged | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-limit5-prune | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-limit5-staged | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-combined5-prune | 64 | 27 | 104 | 912 | 24 |
| gpt-2048-combined5-staged | 64 | 27 | 104 | 912 | 24 |

Per-BM/accumulation counts and every anchor's unpruned universe are in candidate-counts.json.
