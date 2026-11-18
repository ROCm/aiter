# A09: final P256 candidate review and acceptance

Both success conditions remain required: fixed P256 execution and performance
better than, or non-regressing against, the frozen original native. This is a
local candidate. GPU correctness and performance are pending.

## 1. Is 8 B of scales per wave justified?

Keep it as a candidate, with performance evidence still limited. A full task
reads 2048 B of BF16 and writes 1024 B of FP8 plus 32 B of scales: 3104 B in
both layouts. Wave w writes scales 8w through 8w+7, using lanes 0,4,...,28.
The old default-buffer PS also reaches the scale store in every wave. It masks
non-owners with OOB offsets, not EXEC. Valid writer count therefore does not
show a fourfold increase in store instruction issue.

The change splits one wave's 32 valid bytes into four waves' 8 valid bytes.
Effective memory requests, lower-level combining, cache traffic and service
cycles have not been measured. The explicit layout removes 32 B of dynamic
LDS and two TG synchronizations from the old PS. Frozen single-tile native
has one such synchronization. QAT already has no LDS.

## 2. Which implementation is selected, and why?

Use reviewed code `1777a3f1e67220ff0450f46f1d179b6076358733`, based on
production `08e20f6`. It retains A09
`ff4eec11474431671047122be058ad43df5f8be0` and adds buffer access in
`aiter/ops/triton/_gluon_kernels/gfx1250/quant/fp8_group32.py`.
The shared math helper, direct non-PS path, grid `(256,)`, four wave32 waves,
1024 elements/task and task stride 256 remain fixed.

Installed Gluon `amd.cdna3.buffer_load/store` lower to gfx1250 buffer opcodes.
Each tile uses a 64-bit scalar base and local 32-bit offsets. This avoids the
buffer offset limit for large tensors. Global element and group masks remain
in force, and masked loads return zero. QAT explicitly converts the multiply
result to BF16 because buffer_store does not perform that conversion.

The local binder comparison keeps 128-bit loads, 64-bit FP8 stores, byte scale
stores and 128-bit QAT stores. It removes the data-mask EXEC branches. It does
not remove all XACK waits or all differences from native. The resource cost
is explicit:

| PS path, `[513,2048]` binder | Before: global | Selected: buffer |
| --- | ---: | ---: |
| FP8 VGPR / code bytes | 46 / 1388 | 53 / 1436 |
| QAT VGPR / code bytes | 45 / 1352 | 47 / 1420 |
| SGPR | 68 | 68 |
| Dynamic LDS / scratch bytes | 0 / 0 | 0 / 0 |

Removing the opcode and EXEC-mask difference is the selection rationale.
Extra registers, descriptor work and remaining waits can offset that benefit.
Against frozen native, the selected FP8/QAT PS uses 53/47 versus 49/44 VGPR.
Compiler VGPRBlocks remain 3/2 for both. This is an allocation comparison,
not a runtime occupancy result. The selected code is 1436/1420 B versus
native 1388/1264 B. QAT has no LDS-removal benefit; it must pass separately.
This is not a performance result. No barrier or wait was manually deleted.
The local ISA review checks descriptor reuse, store sources, SCC/VCC and loop
backedges. MI400 branch replay rules protect cross-iteration source reuse;
VM_VSRC alone is insufficient. Runtime RETRY_DISABLE remains unknown, and
MODE.REPLAY_MODE=1 does not establish that retry is enabled.

## 3. Are correctness and review complete?

- [x] Codex reviewed the existing A09 kernel, wrapper, tests and compiler,
  including code that was already present before this diff.
- [x] Local builds: 12 native/candidate variants and all 12 frozen hashes.
  Actual offline IR/ISA also covers FP8/QAT, PS/non-PS and tail inputs.
- [x] CPU-only import harness collected the same 45 node IDs. Collection
  executes no numerical tests. Host checks cover task/scale ownership and
  benchmark provider routing with a fake timer.
- [x] Local Black checks (one file per invocation), Ruff and diff whitespace
  checks passed.
- [x] The two existing A09 Graph nodes now also check QAT over 20 changing
  inputs, with preallocated outputs. FP8 checks and all other nodes remain.
- [x] External runner: fixed offline acceptance passed, exit 0. It checked
  12 builds, frozen hashes, style, 45 collected node IDs and four binder paths.
- [x] Claude Opus 5.5 independently reviewed the full implementation and diff;
  verdict `approve`, no blocking findings. This means local readiness only.
  A09-R1/R3: this guide now runs correctness without profiling and uses an
  explicit profiler branch for two Graph nodes. A09-R2: the validation guide
  states that the benchmark default selects A09 FP8/QAT; the handoff uses
  explicit `--ops`. Private artifacts retain the original findings.
- [ ] GPU: all 45 tests pass, zero skips, zero byte or guard mismatches.
  `[513,2048]` has 1026 tasks; two workers run five iterations.
  `[513,2080]` has 1043 tasks; the final task has one valid group.
- [ ] GPU: save actual wrapper JIT and confirm 256 TGs x 128 threads for PS.
- [ ] GPU: pass final native-versus-PS performance acceptance for FP8 and QAT.

A03 `7108dbec079595026f76b7b1b8e082fe1669796d`, frozen native source and its
manifest are unchanged. Public comparison base is
`a8a913883d7be9837c2f46f2e92de8fc701bac26`; A09 native is ATOM
`10b99a085ba46e5b3a125f0114cec7c71e401552`.
The [historical hardware report](mono-results/2026-11-18-gfx1250/README.md)
belongs to `d5930aff35d67fa0a2e74ba425909747f3ccef9f`: 36 tests passed,
eight timing rows regressed, four were inconclusive. It does not validate
this candidate or its QAT changes.

## 4. What is the one hardware handoff?

Follow [the validation guide](mono-persistent-validation.md) after local
acceptance and Claude review. Use the supplied final source and the existing
working ROCm environment. Run correctness and save launch/JIT evidence first.
Then compare only frozen native and selected PS with:

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --ops a09 a09-qat --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json"
```

Keep every round's provider order, latency, absolute difference and timestamps.
Hot/rotating describes input reuse. Outputs are allocated by the wrappers
when each graph is captured; there is no shared preallocated output pool.
Regress if the PS median increase exceeds native P90-P10 and PS P10 exceeds
native P90. Pass a row only if the median decrease exceeds native P90-P10
and PS P90 is below native P10. All other rows are inconclusive. This strict
rule does not treat equal medians or overlapping ranges as equivalence.
Every required row must pass for that path to pass; there is no positive
slowdown allowance. The validation guide states the complete gates.
No automatic rerun or tuning sweep follows an inconclusive or failed result.
Return it with the evidence. Only beating old PS cannot satisfy acceptance.
