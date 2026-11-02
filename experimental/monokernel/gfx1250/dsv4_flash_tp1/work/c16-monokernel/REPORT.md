# C16 monokernel region study

Implemented a single-launch C16 layer in `model_layer_batch.py`; native serving dispatch remains unchanged. This is the DeepSeek V4 Flash TP1 design on one gfx1250 GPU per experiment. Four GPUs ran independent workers. It is not Pro TP4 or an E2E throughput benchmark.

All values below are microseconds per decode batch, not per token. Shapes are decode position1023 (1K prompt), with128 SWA,384 CSA,136 HCA KV entries. Compressed-layer tests include the1023 compression boundary. This samples the start of a1K-input/1K-output workload, not its entire1024-token decode trajectory.

Native columns run installed ATOM/AITER APIs with real checkpoint projection/expert weights where applicable. Native kernels are listed verbatim in KERNEL_NAMES.json and profiler traces. Neither native column is a vector-control implementation. Each paired microbenchmark uses32 operations in a flat HIP graph,5 warmups,40 alternating timing samples,2 rounds; the reported value is the second-round median. C8 and C16 counterparts use the same GPU for each region family except the final QKV-cache run (GPU0/GPU1); each native/fused pair shares one GPU.

Fused columns benchmark isolated wrappers around the implementation bodies used by the layer, not subtraction of prefix times. Compiler register/LDS allocation and entry/exit synchronization differ from the full layer. Subregion times must not be summed to claim whole-layer/E2E latency. The whole-FFN row is measured as one fused region against its native call chain using identical inputs and checkpoint weights; it overlaps the four FFN subrows. Its learned gate differs from the routed-only row's gate, so its expert union/cache behavior can also differ.

| Region | Native C8 µs | MK C8 µs | Native C16 µs | MK C16 µs |
|---|---:|---:|---:|---:|
| Attention mHC pre + RMSNorm | 8.52 | 6.95 | 8.62 | 7.19 |
| CSA compressor: projection + pool/norm/RoPE + state/cache write | 10.37 | 9.80 | 10.49 | 9.89 |
| CSA indexer compressor: projection + pool/norm/RoPE/quant + state/cache write | 7.21 | 7.99 | 7.39 | 8.56 |
| CSA identity top-k + index packing (256 compressed tokens) | 2.57 | 1.35 | 2.63 | 1.36 |
| HCA compressor: projection + pool/norm/RoPE + state/cache write | 11.20 | 15.92 | 10.89 | 15.59 |
| Input quant + QKVA + norms/RoPE + QB + KV-cache write | 27.72 | 18.35 | 13.08 | 18.38 |
| SWA attention + split-KV reduction (128 KV) | 7.49 | 8.08 | 10.95 | 10.44 |
| CSA attention + split-KV reduction (384 KV) | 9.56 | 9.40 | 12.53 | 13.02 |
| HCA attention + split-KV reduction (136 KV) | 7.93 | 8.29 | 11.47 | 10.80 |
| Inverse RoPE + quant + OA/OB projections | 63.68 | 24.13 | 15.77 | 29.41 |
| Attention mHC post + FFN mHC pre/RMSNorm | 5.31 | 6.69 | 5.46 | 6.74 |
| Router GEMM + shared expert (up/activation/quant/down) | 51.42 | 17.73 | 17.02 | 18.65 |
| Expert selection + routing metadata | 7.86 | 6.33 | 9.74 | 9.85 |
| Routed MoE: input quant + up/SwiGLU/quant + down/reduce | 48.90 | 56.23 | 80.09 | 89.64 |
| Shared+routed sum + final mHC post | 4.96 | 3.24 | 5.05 | 4.28 |
| Whole FFN region (overlaps previous four rows) | 153.67 | 82.30 | 111.63 | 123.38 |

C16 design uses16 live WMMA rows,256 resident CTAs, four attention KV splits (C8 uses eight), batch-sized shared-expert scale strides, and loops covering all512 final residual/post tasks. Retained optimizations: producer-local QKV input quantization; bounded L2 weight prefetch; early QB and OA/OB TDM; AITER mHC producers/reducers; direct current-KV use; fused input/output residual layouts. Fine-grained experimental tile mailboxes remain disabled and are guarded against C16 offset overlap.

Native HERD switches on atC16. The C16 route implements top7→batch popularity→drop least popular/lowest score/lowest index→renormalize6. Drop decisions have dedicated scratch, never the subsequent GEMM gather map. The two new internal HERD barriers have only16 participants, allowing other CTAs to proceed with independent shared-expert work. Fresh coherent reads prevent reuse of the pre-drop metadata.64 random/bias/tie/replay checks match native histogram, gather/scatter and weights exactly.

Four KV splits reduced C16 attention from13.56→10.44µs SWA,16.08→13.02µs CSA,13.72→10.80µs HCA. The biggest remaining C16 gaps are OA/OB, QKV, and boundary HCA compression. C16 native GEMM dispatch changes substantially: C8 logs show untuned shapes falling back to FlyDSL; C16 selects different tuned kernels. Thus native C16 may be faster in absolute time than native C8. The JSON/profiler traces preserve that evidence.

Numerical validation: QKV Q relative RMS<=4.35e-8 and cache exact; OA/OB and shared outputs exact; routed-only MoE and whole FFN final outputs exact in their checkpoint fixtures. mHC and split attention are not universally bitwise native-equivalent: worst selected attention relative RMS≈0.002; mHC post/pre normalized output<=3.86e-5.64 routing checks are separate. No new GSM8K or serving quality run.

All three complete C16 layer variants pass finite-output, thirteen intermediate optimized/control comparisons, and repeated graph replay checks. They compile with418–428 VGPRs,157156B LDS and zero spills. C8 equivalents use415–417 VGPRs. Full layer requests4 warps and compiles to8 physical waves due to mHC warp specialization. Actual-body standalone QKV extraction reports2 spills atC16 even though the full layer has none; that diagnostic and its timings are retained in groups-16-*.json, separate from the checkpoint-backed isolated table above.

Full-layer synthetic-fixture timings (not native comparisons): C8 SWA/CSA/HCA239.72/281.35/292.42µs; C16 313.68/354.98/370.39µs. They use synthetic expert weights and inputs and must not be presented as native checkpoint A/B data.

Reproduction: run `native_worker.sh 0..3` on separate HIP_VISIBLE_DEVICES, then `TEST_BATCHES=16 python final_swa.py` and `TEST_BATCHES=16 RATIO=4|128 python final_compressed.py`. Full harnesses include validation and two-round timing. `check_herd.py` runs64 native routing checks. The container must provide/app/ATOM,/app/aiter,/data/DeepSeek-V4-Flash and/work mounted to the restored dsv4-monokernel tree. See RESTORE.md for the complete bundle.
