# WMMA QKV region — Flash C1

Implemented inside the one-launch Q-A/KV → normalization → Q-LoRA quantization → Q-B → per-head Q normalization → RoPE region. This is the isolated layer-0 prototype, not ATOM serving, Pro TP4, or a new full-model benchmark. Other regions retain their existing implementations.

Accepted arithmetic: load original packed FP8 checkpoint weights and E8M0 scales, dequantize exactly representable FP8 values to BF16 for WMMA operands, accumulate in FP32. This uses BF16 WMMA, not vector dot products and not FP8 WMMA. Quantization boundaries remain FP8. ISA confirms `v_wmma_f32_16x16x32_bf16`.

| Same-run arm | GPU time, µs | Launches |
|---|---:|---:|
| Unfused vector split control | 27.475 | 6 |
| Previous fused vector region | 29.023 | 1 |
| New fused BF16 WMMA region | 22.613 | 1 |

Latency reduction: 22.08% versus fused vector; 17.70% versus vector split control.

Timing: second round only, median of 30 profiler-observed graph replays. Split time sums six kernel durations per replay. Weights and region inputs are reused, with no restoration copies in timed replays: the region overwrites its internal buffers. Compare these same-run arms rather than the older region table, which restored all workspace buffers before every replay. No native AITER timing claim.

Design:
- Grid: 256 CTAs, 4 warps / 128 threads each. All CTAs participate in the six internal semaphore barriers.
- Pack input once: 32 CTAs each handle 128 elements, writing packed FP8 plus E8M0 activation scales. This extra input-pack stage adapts the old prototype FP32 dequantized-input ABI.
- Q-A: 32 output channels per tile × 8 K splits = 256 CTA tasks, K=512 per task. KV: 16 output tiles × 8 splits = 128 CTA tasks. These calls remain sequential within each CTA.
- Q-A/KV partials: FP32 global workspaces, 8192 + 4096 elements (48 KiB). CTA 0 reduces each eight-way partial result, rounds the projection to BF16, then applies weighted RMSNorm. Q and KV norms remain sequential.
- Q-LoRA quantization: 8 CTAs write packed FP8, E8M0 scales, and the original dequantized diagnostic buffer.
- Q-B: 128 output channels per tile, 256 CTA tasks, full K=1024 per task. WMMA uses a 16-row tile with only token row 0 active.
- Per-head Q norm: 64 CTAs. Q RoPE: 64 CTAs; KV RoPE: CTA 0. These retain the original separate stages.
- Payload uses explicit device-scope buffer loads/stores; existing bounded semaphore protocol and monotonically advancing epochs are preserved. Packed buffers and partials are allocated once outside capture and overwritten before consumption each call.
- Compiled resources: 228 VGPRs, 32768 bytes compiler-reported shared memory, zero SGPR/VGPR spills, zero private scratch. Grid stays within the GPU CU count.

Validation:
- Token 42 initial test plus eight additional real embedding rows: 0, 1, 7, 43, 100, 1024, 8192, 32000. Same checkpoint weights and controlled region inputs as the prototype.
- Worst additional-embedding relative RMS error against independent FP32 reference: 0.000810%. This is numerical agreement, not bitwise parity or native ATOM validation.
- Independent raw projection checks, 1000 changing-input graph replays, signed/unsigned counter wrap, zero health errors, and one regional kernel per replay.
- The current 0.5% diagnostic gate is preserved; the accepted BF16 WMMA variant additionally passes a 0.01% independent-reference gate for all tested intermediate buffers.

Rejected experiments:
- Software-scaled FP8 WMMA: ~79.86 µs.
- Hardware-scaled FP8 WMMA without split-K: ~61.03 µs.
- Hardware-scaled FP8 WMMA with split-K: ~22.5 µs, but token 1 reaches ~0.70% error at Q-LoRA quantization and ~0.68% after Q RoPE; rejected by the original 0.5% diagnostic gate. BF16 WMMA reduces this discrepancy with comparable latency. This establishes an arithmetic-path difference; it does not isolate the precise hardware rounding rule.
- Rejected source/log/ISA/results are preserved in sibling wmma-v1-software-scales/, wmma-v2-scaled/, wmma-v3-splitk/, and wmma-scaled-diagnostic/.

Reproduce (GPU0 must be idle):
```sh
docker start vigokhal_dsv4_tp4
docker exec -e HIP_VISIBLE_DEVICES=0 -e TRITON_CACHE_DIR=/work/full-layer-c1/triton-cache vigokhal_dsv4_tp4 bash -lc "ulimit -c 0; cd /work/full-layer-c1/regions/wmma; python test.py --grid 256"
```
Copy results first: the harness overwrites fixed filenames. setup.py loads checkpoint weights from /data/DeepSeek-V4-Flash. No model weights are included in the archive. No server, published branch, or image was modified.
