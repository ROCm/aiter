# Resume here

Read `REPORT.md` for conclusions, measurements, accuracy failures, and scope.

- `accepted.py`: supported BF16-WMMA fused launcher; default split-K=16. Uses `fused.py` with rejected native-FP8 paths disabled.
- `opt.py`: optimized projection plus separate reduction/norm alternative.
- `baseline/kernels.py`: original isolated controlled A/B implementations.
- `fixture.pt`: layer-0 projection weights, scales, norm weights, live input, and nine quantized input cases.
- `layer-fixtures/`: corresponding Q-A/KV weights for all 43 Flash layers.
- `final-benchmark/`: final two-round samples, generated assembly/LLVM, resource metadata and profiler traces.
- `final-validation/`, `layer-validation/`: numerical checks; `layer-sequence-bench/`: isolated regions iterating actual weights from all 43 layers.
- `sweep*/`, logs, code snapshots, `triton-cache/`: full experiment record, including rejected variants.

## Reproduce on this machine

The existing container `vigokhal_dsv4_tp4` has the parent directory mounted at `/work`:

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 \
  bash /work/qkv-asm-opt/run.sh
```

Check GPU availability before running. The script measures twice and retains both rounds; report round 2. Rerunning overwrites corresponding result files, so copy the directory first if preserving the original measurements.

## Reproduce after extraction elsewhere

Use gfx1250 and the matching software environment. The tested container was created from `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`; exact image and Python package metadata are recorded with the results. Mount this extracted directory into that environment and run `bash run.sh` there. The benchmark and saved-fixture validations need **no full `/data` checkpoint**, serving server, or ATOM checkout.

To regenerate the 43 layer fixtures from the original model, set `USE_CHECKPOINT=1 MODEL_ROOT=/data/DeepSeek-V4-Flash` when running `validate_layers.py`.

## Kernel ABI and ownership

X: contiguous FP8 E4M3FN[4096]; XS: uint8 E8M0[32]. W: contiguous FP8 E4M3FN[1536,4096], Q-A rows followed by KV rows. WS: uint8 E8M0[12,32], block scale 128×128. NW: FP32[1536]. Output: FP32[1536], first 1024 Q latent and last 512 KV latent, each independently RMS-normalized after BF16 rounding, epsilon 1e-6.

Create `QAKVNorm(device='cuda', split=16)` outside capture, warm it once, then call `state(X, XS, W, WS, NW, out=optional_output)`. One state owns its accumulator and two padded counters. Calls sharing a state must be sequential on one stream; use independent states for overlapping executions. Returned default output is reused on subsequent calls. All per-call resets occur inside the kernel. No pointer/shape validation or multi-stream arbitration is performed on the fast path.

This is an isolated Flash projection/norm artifact. It is not integrated into ATOM and does not implement Q-B, RoPE, attention, or MoE. Do not use timings as tokens/s or claim an accepted full-layer monokernel.
