# Flash live vector experiment — not accepted for serving

This is a **partial ATOM integration** of the unfused vector control, not the WMMA prototype and not a whole-model monokernel. Layers 0 and 1 execute the split vector layer during C1 decode, including attention and MoE. The other 41 layers, prefill, and other concurrency sizes use native ATOM. Raw checkpoint weights are duplicated for the two experimental layers; this is a prototype memory cost.

The live numerical check failed: 1.984% relative RMS versus corrected native ATOM, 1.009% versus independent PyTorch equations, with a 0.1% acceptance threshold. Any performance measurements are **diagnostic only**, not a correctness-qualified speedup. No GSM8K evaluation was performed in this experiment.

`ATOM/` is a complete isolated copy of the original image checkout plus this experiment. Original `/app/ATOM` remains unchanged. `integration.patch` contains only our changes to existing model and MoE files; the added package is `ATOM/atom/models/dsv4_vector/`. `ROUTING_FINDING.md` explains the image-specific decode hash-routing defect and the correction applied to both paired paths.

## Reproduce on the existing container

Image: `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`.
Host directory `/home/vigokhal/dsv4-monokernel` must be mounted at `/work`; Flash checkpoint at `/data/DeepSeek-V4-Flash`. Use GPU 0 alone.

Launch **one** server per configuration:

```bash
# Original image baseline:
docker exec -d vigokhal_dsv4_tp4 bash /work/live-vector/baseline.sh
# OR corrected native baseline:
docker exec -d vigokhal_dsv4_tp4 bash /work/live-vector/hash-baseline.sh
# OR experimental vector candidate (numerical checks disabled for diagnostic timing):
docker exec -d vigokhal_dsv4_tp4 bash /work/live-vector/candidate.sh
```

After `/health` is ready, run the two clients without restarting the server:

```bash
docker exec vigokhal_dsv4_tp4 bash /work/live-vector/paired-clients.sh live_vector_candidate
```

Change the result prefix for each configuration. Report **only run2**. Workload: single GPU, concurrency 1, four measured requests, 1024 input and 1024 output tokens per request, seed 0, ignore EOS, prefix cache disabled. Each entire client invocation also contains its own initial test and warmup. Total token throughput counts both input and output tokens.

Stop only this experiment's server between configurations. Never start overlapping GPU servers. The launcher `validation.sh` uses eager execution and dual evaluation; its timings are not benchmark results. It records native differences and runs the independent reference check, which currently aborts at the documented acceptance failure.

## Files

- `acceptance.json`: explicit failed acceptance status.
- `reference-layer0/reference-comparison.json`: per-stage independent comparison.
- `live-checks.jsonl`, `validation-server.log`: live native comparison evidence.
- `debug-live.pt`: live intermediate tensors for debugging.
- `baseline-server.log`, `hash-baseline-server.log`, `candidate-server.log`: configuration logs.
- `*-client1.log`, `*-client2.log`: whole-client logs.
- `*-server-before.txt`, `*-server-after.txt`: process identity evidence.
- Raw benchmark JSONs are in adjacent `../flash/results/`, copied into `results/` for the handoff.

## Resume priorities

1. Fix live numerical drift before claiming a serving speedup. The independent comparison first develops small attention reduction/rounding differences that expand across repeated FP8 quantization boundaries; this is an observation, not yet a proven complete diagnosis.
2. Validate both experimental layers over varied inputs and long/wrapped cache windows; layer 0 failed first, so layer 1 has not passed equivalent validation.
3. Keep native hash routing correct in both sides of subsequent comparisons. The earlier image baseline numbers used the uncorrected image.
4. After correctness, expand support beyond the two uncompressed hash layers. Compressed/indexer layers and non-hash routing remain native.
5. Do not substitute the all-WMMA prototype silently: it has a separate failed acceptance status in `../full-layer-c1/all-wmma/`.
