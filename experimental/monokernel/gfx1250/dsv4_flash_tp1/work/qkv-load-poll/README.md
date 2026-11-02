# Fully fused Flash C1 Q/KV preparation with load polling

Default callable: `region.QKVRegion(W,WS,NW,QB,QBS,mode='single')`.
Then `q,kv = state(X,Cos,Sin)`. See its docstring for tensor ABI. `mode='two'` is the matched control.

One GPU launch includes input FP8 quant, QA/KV WMMA, fixed-order split16 reduction, weighted norms, latent Q quant, QB WMMA, per-head Q norm and Q/KV RoPE. No attention, KVcache write, MoE, serving integration, branch push or image change.

The accepted version deliberately uses fixed-order FP32 scratch reduction rather than projection atomic adds. The latter has a newly reproduced FP8 rounding-threshold failure; older atomic-based timing results are not accepted quality evidence for the extended chain. Read REPORT.md.

Requirements: gfx1250, >=256CUs, existing image rocm/fw-bringup-mlperf:gfx1250-atom-20261015. No concurrent GPUwork while running the resident grid. One state per stream, sequential invocations only. Outputs are reused by later calls. After synchronization, nonzero `state.health` means INVALID output (watchdog timeout). State allocation is outside measured latency, all invocation state updates inside.

All required layer0 model weights and inputs are in fixture.pt; full model checkpoint not needed. Existing host directory is mounted at /work/qkv-load-poll in vigokhal_dsv4_tp4. Check for other GPUusers first. Reproduce with:

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 bash /work/qkv-load-poll/run.sh
```

Reruns overwrite result files; copy the directory to preserve recorded artifacts. Environment versions and exact ISA/LLVM are under scratch-benchmark/. Report only timing round2.

Key files:
- region.py: default one-launch API, matched two-launch control.
- combined.py: device-scope load polling and grid/Qready synchronization.
- qa.py: fixed-order and experimental atomic split-K modes.
- qb.py: N128/Kunroll4 WMMA plus headnorm/RoPE.
- scratch-validation/, scratch-stress/, scratch-benchmark/: accepted candidate evidence.
- publication-acyclic/:10240publications perstyle,255consumers and512values perpublication.
- wrap.json: uint32 barrier/semaphore counter-wrap cases.
- variants.py: all experiment definitions. Names containing atomic refer to source-level add-zero polling; compiler actually emits loads. QAMODE1 is the numerically rejected data-atomic projection mode, independent of polling style.
- final-validation/, final-benchmark/, stress/: initial atomic-projection exploration, not accepted results.
- rounding-diagnostic/: confirms old two-launch and new single-launch failures match exact current latent quantization; no visibility error in reproduced failures.
- publication.log/publication-diagnostic.log: initial100ms watchdog failures in long cyclic handshakes.
- publication-long.log: longer-bound cyclic test completed add-zero-control and scalarRT passes; HT test was interrupted after prolonged execution, vector style not reached. This test is not claimed to have passed all styles.

Do not run publication_long.py or publication_diagnostic.py as the default verification path. They are retained failed/incomplete experiments. The accepted acyclic test matches one publication per dispatch, with changing payloads across graph launches. HT polling is not the default.
