# Q-A/KV FP32 atomic ablation — gfx1250 C1

**Best atomic: 5.144 us; baseline: 4.581 us. Atomic variant is 12.3% slower.**

Scope: Flash layer0 Q-A[1024,4096] + KV[512,4096], weighted RMSNorm over1024 Q latent and512 KV elements. No Q-B, RoPE, attention, MoE, or input quantization timing. All variants use identical BF16 WMMA operands from packed FP8/E8M0 checkpoint weights, FP32 accumulators, BN32, splitK8,384 producer CTAs and4waves/CTA. BF16 rounding before norm is preserved.

Second full timing round only: median of30 samples, alternating variant order,100 complete invocations per graph sample. First round discarded. Initialization is included every invocation in B–F. G resets state inside its last consumer; all per-invocation reset work is timed. G requires one initial zero before first use. Harness performs that zero outside each100-invocation timing batch because variants share scratch; no per-invocation initialization is excluded.

| Variant | Launches | Round2 us | Versus A |
|---|---:|---:|---:|
| A: scratch + separate reduction/norm | 2 | 4.581 | +0.0% |
| B: atomic + fused norm, acquire/release | 2 | 8.916 | +94.6% |
| C: atomic + separate norm | 3 | 5.265 | +14.9% |
| D: scratch + fused norm, acquire/release | 2 | 9.096 | +98.6% |
| E: atomic + fused norm, explicit DEV completion | 2 | 6.116 | +33.5% |
| F: scratch + fused norm, explicit DEV completion | 2 | 6.842 | +49.4% |
| G: E with in-kernel reset for next invocation | 1 | 5.144 | +12.3% |

## Interpretation

FP32 atomics help versus scratch under the matched explicit-completion fused protocol: E6.116us vsF6.842us. However, both lose to A. Two completion handoffs remain: eight splits produce each tile, then32 Q tiles or16 KV tiles finish each normalization group. There are no spinning CTAs or grid barriers.

Acquire/release counters emit `global_wb scope:SCOPE_DEV`. Explicit DEV-scoped payload loads/stores, completion waits and relaxed counters eliminate those generated writebacks: B8.916 -> E6.116us. This does not imply the device-scope writeback necessarily goes to HBM. In-kernel reset then saves another launch: E6.116 -> G5.144us.

ISA: G uses130VGPR,8KiB LDS,zero SGPR/VGPR spills and zero private scratch. Other producer variants use128VGPR with the same LDS/spill status. BF16 WMMA and DEV-scoped FP32 atomic instructions are verified; G has no global_wb/global_inv or system-scope instructions.

## Correctness

All seven variants passed nine input cases: one actual live-serving projection input and eight checkpoint embedding-derived normalized inputs. Independent reference uses FP64 projection, the same BF16 output boundary, then FP32 RMSNorm. Maximum relative RMS error versus A or independent reference:7.2942e-9, below the unchanged0.1%gate. Scratch poisoning and completion counters were checked.

Each fused variant B/D/E/F/G completed1000 changing-input graph replays. Outputs were checked every100th iteration and at the end; worst checked relative RMS versus A7.2942e-9. This validates only this isolated ablation, not the previously failed full-layer/ATOM experiment.

## Reproduce and artifacts

Original image: rocm/fw-bringup-mlperf:gfx1250-atom-20261015. Mount parent directory at /work, checkpoint at /data/DeepSeek-V4-Flash. GPU0 must be uncontended. No server is used.

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 -e TRITON_CACHE_DIR=/work/qkv-atomic-ab/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; cd /work/qkv-atomic-ab; python bench.py > rerun.log 2>&1'
```

timings.json retains both rounds/all samples. Separate profiler passes are in *-trace.json and are not substituted for graph elapsed timings. correctness.json, stress.json, resources.json, ISA, live-input.pt and source hashes are included. Prior experiments remain in logs/backups. Final table uses only the final run’s round2.
