# Actual ATOM QKV C1 microbenchmark

This harness calls the installed ATOM `LinearBase.forward`, `LinearBase.process_weights_after_loading`, `RMSNorm.forward`, and `qk_norm_rope_maybe_quant`. Native kernel names were independently matched to live serving graph bs_1_q_1. There are five native launches; input quant, QA/KV GEMM, Q-latent RMSNorm+quant, QB GEMM, fused Q/KV norm+RoPE+SWA cache write. No handwritten native replacements.

Layer0 Flash weights from the existing checkpoint fixture, BF16 input[1,4096], FP8 QA/KV weight[1536,4096], FP8 QB weight[32768,1024], Q[1,64,512], KV[1,512]. Decode position1024 uses ATOM actual BF16 RoPE cache; validation also covers1536,2047. Native weight preshuffle enabled, mxscale_shuffle disabled, both scales E8M0; no tensorparallel distributed arithmetic at C1/TP1. Singleton process group metadata is stubbed during module construction only.

## Measurements

GPU2; reused weights; graphs contain128 calls,5 warmups,40 samples alternating case order per round. GPUevent median elapsed per invocation. Report round2 only; both rounds and raw samples in results.json. These are complete region graph times, not an arbitrary sum of independently timed operations or serving tok/s.

| Region boundary | Actual ATOM (us) | Prototype (us) |
|---|---:|---:|
| Input quant through Q/KV RoPE **and cache write** | 26.5413 (5launches) | 12.5309 (complete + cache_store,2launches) |
| Without cache write (diagnostic, altered native call) | 26.1624 | 11.5260 (complete,1launch) |

## Native exact profiler names

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `gemm_a8w8_mx128_t16x512x256_mw1_nw4_nb2_sk1_cm1_cn1`
- `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDB8_Li128ELi8ELb0ELb1ELb1ELi1ELb0EEEvPT0_PT_PfS5_S5_S5_dbiiiiiiibb`
- `gemm_a8w8_mx128_t16x1024x128_mw1_nw4_nb2_sk1_cm1_cn1`
- `qk_norm_rope_H64_D512_RD64_kvw_r4_w32_flydsl`

## Numerical status

**Not yet a native-equivalent replacement at the previous0.001 relative-RMS gate.** Extended validation failed216/222 cases; maxQ=0.009270032, maxKV=0.001813752. All barrier health flags zero. Initial fixture only passed narrowly; broader tests use live fixture,9 fixture inputs,64 seeded random inputs ×3 positions.

Four diagnostic inputs have identical native/prototype input FP8 bytes and scales. Pre-normalization QA GEMM results differ relativeRMS0.00047–0.00097; this changes6–12 latent FP8 elements and amplifies downstream QB/Q differences. When both norm+quant implementations receive the exact same native QA result, FP8 outputs and scales match exactly in these four cases. Evidence points to differing GEMM accumulation/split order (native QA split1; prototype split16), not a norm/quant input mismatch. No prototype changes were made in this task.

Profiler traces serve kernel-name verification only; their one-off durations are not the reported benchmark timing.

## Reproduce

`docker exec -e HIP_VISIBLE_DEVICES=2 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/atom-actual-microbench/qkv; python bench.py && python validate.py && python diagnose.py'`

`native.py` exposes `NativeQKV(cache=...,dest=0,positions=1024)`; calling with BF16x returns(q,kv), writescache, and preserves actual call path. Avoid overlapping invocations sharing buffers. Changing positions can be done with `n.pos.fill_(p)` between uncaptured runs.
