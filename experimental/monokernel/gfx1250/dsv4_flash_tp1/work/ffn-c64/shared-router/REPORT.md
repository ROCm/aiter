# Flash C64 shared expert and router ablation

Actual ATOM methods: `Expert.forward`, `MergedColumnParallelLinear` and `RowParallelLinear` inherited `LinearBase.forward`, native HIP quant and clamp-SiLU×up. TP1;64 true token rows (no looping C1 kernels). The 1024-input/1024-output workload gives these FFN shapes at every decode position; sequence length does not alter these dense matrices.

Checkpoint layer0 shared weights are FP8 E4M3 per128 block E8M0 scales: gate and up each[2048,4096], merged[4096,4096]; down[4096,2048]. Inputs BF16[64,4096]. Intermediate BF16[64,4096], activation BF16[64,2048]. SwiGLU clamps gate≤10 and up to[-10,10]. Baseline uses the original script defaults (ATOM_V4_USE_TRITON_FUSION=0), so its quant stages remain separate.

Router weights layer0 and layer3 are BF16[256,4096], output logits BF16[64,256]. Layer0 is configured as hash-routed and layer3 as learned, but the actual C64 HERD path bypasses `MoE._hash_topk`; serving capture contains HERD routing, not the hash_topk kernel. Therefore the router timings here cover only the BF16 projection. Actual HERD selection/pruning is benchmarked by the routed-expert harness; no hash-hook timings are substituted.

## Second-round results

GPU2;128 repetitions using flat HIP graph kernel-node clones and identical buffer addresses,5 warmups,40 alternating-order samples per round; GPUevent medians in microseconds. Both rounds and raw samples are saved in timings.json. These are isolated ablations; do not add individual timings to predict chained execution.

| Operation | Actual ATOM µs | Existing candidate ablation µs |
|---|---:|---:|
| Input per128 FP8 quant | 1.7985 | — |
| Shared gate/up GEMM | 4.9196 | 4.9081 |
| Clamped SiLU × up | 1.5134 | — |
| Down-input per128 FP8 quant | 1.7742 | — |
| Shared down GEMM | 3.4035 | 3.2697 |
| Entire shared expert chain | 16.4759 | 18.3598 |
| Router GEMM layer0 | 3.6764 | — |
| Router GEMM layer3 | 3.6944 | — |

A dash means no separate existing M64 custom implementation, not zero latency. Both shared chains contain five launches. The candidate reuses the existing public AITER Gluon projection path with the earlier selected M8 tile configuration at M64: BM32/BN32/BK256,4warps,3buffers,TDM,splitK1. The kernel supports M64 and was validated here, but the saved serving dispatch configuration has **no M64 shared-expert keys**. Thus this is an explicitly invoked M64 ablation, not a claim that the prior branch automatically selects it. No new kernel or full M64 monokernel was written.

The candidate chain is slower than native despite similar isolated GEMM times. Preserve the native chain; isolated GEMM differences are not sufficient to claim a serving improvement.

Validation:8 seeds ×64 token rows, real checkpoint weights, synthetic normally distributed BF16 inputs. Up, down and complete shared-expert outputs are bitwise equal to actual ATOM in all8 cases (relative RMS0). This is microbenchmark validation, not model accuracy/GSM8K.

## Kernel names

### native_shared_chain

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `gemm_a8w8_mx128_t64x32x512_mw4_nw1_nb4_sk1_cm1_cn1`
- `_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_if`
- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `gemm_a8w8_mx128_t32x32x512_mw2_nw2_nb3_sk1_cm1_cn1`
### candidate_shared_chain

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`
- `_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_if`
- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`
### native_router_layer0

- `_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_16_BLOCK_K_512_NUM_BUFFERS_4_LAYOUT_TN_USE_ACTIVATION_0_ADD_BIAS_0`

## Reproduce

```sh
docker exec -e HIP_VISIBLE_DEVICES=2 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/ffn-c64/shared-router; python bench.py'
```

`native.py` provides NativeSharedRouter; `fixture.pt` stores layer0 shared weights, layer0/layer3 router weights, layer3 bias, and layer0 hash table. Hash and generic learned selector helpers are for inspection only; neither is included in the primary measured serving table. The graph helper dependency is `/work/atom-actual-microbench/attention/graphbatch.py`.
