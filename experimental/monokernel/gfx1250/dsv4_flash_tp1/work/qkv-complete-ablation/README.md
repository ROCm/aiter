# Flash C1 complete Q/KV preparation ablation

Read REPORT.md for results and scope. All required model weights and inputs are in fixture.pt. No full checkpoint is needed to rerun. No ATOM serving integration or image/branch change is included.

Run on an idle gfx1250 GPU in the existing ROCm image:

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 bash /work/qkv-complete-ablation/run.sh
```

Container image: rocm/fw-bringup-mlperf:gfx1250-atom-20261015. Host directory is mounted at /work/qkv-complete-ablation. Environment versions are in final-benchmark/environment.json. Reruns overwrite output directories, so copy this directory first to preserve the recorded results. Check for other GPU users before running. Single-launch version requires256 available CUs; one state per stream, never overlap invocations sharing state.

- region.py: standalone QKVRegion callable, mode='two' recommended; mode='single' full fused alternative. Its docstring defines the tensor ABI. Return tensors are reused by subsequent calls; copy if retention is needed. For the single-launch version inspect state.health after synchronization; nonzero indicates grid-barrier timeout and invalid output.
- qa.py: input quant, QA/KV WMMA, atomic split16, weighted norms, latent quant and KV RoPE.
- qb.py + tail.py: QB WMMA, head completion counters, Q norm/RoPE.
- combined.py: one-launch variant with bounded device-scope barrier.
- base/opt.py: optimized separate projection/reduction baseline.
- harness.py: fixture loader, benchmark state, independent FP64 reference.
- validate_final.py:268inputs x3replays x3variants, changing rotary angles.
- stress.py:10240checked changing-input invocations per fused variant.
- check_api.py: standalone callable + CUDA graph check.
- final_bench.py:2rounds,128invocations/capturedgraph,40interleaved timing samples; one and16weight copies. Report round2 only. Stage timings measured separately and not additive to complete-chain timing.
- fixture.py: optional regeneration from original checkpoint and earlier QA fixture, NOT needed for normal reproduction.
- exploratory/: generator source snapshots; do not use to regenerate final sources. In particular subsequent direct fixes are authoritative in qa.py/qb.py/combined.py.

fixture.pt contains layer0 Flash weights, saved live input and8reconstructed quantized bank inputs, plus a controlled rotary table (position1024/base10000). Rotary operation is tested against the same supplied table, including changed positions during validation. This is not a claim that this table is the live model's YaRN table.
