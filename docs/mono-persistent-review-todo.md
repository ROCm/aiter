# A09: final P256 candidate review and acceptance

Both success conditions remain required: fixed P256 execution and performance
better than, or non-regressing against, frozen original native. The selected
candidate passes worker local checks. GPU correctness, actual JIT/launch and
performance remain UNVERIFIED. Independent Claude review is pending.

## 1. Which arithmetic change is selected, and is it exact?

Select the exponent-code reciprocal in the shared Gluon tile helper. For
example, code 128 gives scale 2 and exact reciprocal 0.5. For finite BF16 group
maxima with floor 1e-10, codes are 85..247. Scale is `2^(code-127)` and its
reciprocal is `2^(127-code)`; both are normal FP32 numbers. Construct reciprocal
bits as `max(254-code, 0) << 23`, with `0xff800000` for code 256. Replace only
`x / scale` with `x * reciprocal_scale`. This is an exact power of two, not an
approximate reciprocal. The original scale calculation and E8M0 code store
remain unchanged. Keep maximum(-448), minimum(448), then FP8 conversion. QAT
still converts FP8 to FP32, multiplies by the original scale and converts to BF16.

BF16 to FP32 is exact. Division and multiplication have the same exact result
before rounding for these finite scales. The actual compiler keeps FP32
nearest-even and gradual underflow: LLIR has `denormal-fp-math-f32=ieee`,
metadata has `allow_flush_denorm=false`, and ISA has round mode 0 and denorm
mode 3. Native and candidate MODE writes do not change those FP fields.
MI400 Shader Programming Guide section 4.6.2.4 defines these modes. Thus tiny
results, signed zero and overflow retain the same rounding. The new multiply
has no fast-math flags. No compiler or flush mode was changed.

Special values are part of the contract:

| Code / scale / reciprocal | Division and multiplication before clamp |
| --- | --- |
| 85..247 / positive finite / positive finite | Same rounded finite values and signed zeros; signed Inf stays Inf; NaNs remain NaN |
| 255 / +Inf / +0 | Finite inputs give signed zero; Inf and NaN inputs give NaN |
| 256 / -0 / -Inf | Nonzero finite or Inf inputs give opposite-sign Inf; zero and NaN inputs give NaN |

Code 256 defensively covers a surviving NaN in scale coding. It remains even
though GPU generation of that code is unproven. CPU fmax can propagate a
signalling NaN. Code bytes remain 255 and 0 for codes 255 and 256. Gluon's
actual default is `PropagateNan.NONE`. LLIR uses maxnum/minnum and ISA uses
med3_num with the original operand order. Arithmetic quiets signalling NaNs;
the lower clamp selects -448 for an arithmetic NaN. Pre-clamp payload changes
therefore cannot enter FP8. QAT then sees identical FP8 and original-scale
operands, including Inf and signed zero. GPU byte comparisons remain required.

The worker ran a byte-identical private copy of the independent CPU checker:
65,536 BF16 patterns x 165 codes = 10,813,440 pairs. Pre-clamp bits match with
NaNs compared as both NaN; final FP8 and QAT have zero byte mismatches. This is
CPU algebra/reference evidence, not GPU proof. The existing boundary node now
adds deterministic GPU bit fixtures with all BF16 patterns and mixed groups.
It checks both PS/non-PS paths against frozen native with exact FP8/E8M0/QAT
bytes, XOR-prefilled outputs, unchanged input and input/output guards. All
finite-only oracle assertions remain; the new nonfinite data skips that oracle.

## 2. Does the static reduction justify this candidate?

Yes for local selection; runtime speed is still unknown. The actual selected
source, hashes, production HEAD and patch baseline are in the
[validation guide](mono-persistent-validation.md#source-and-local-status).
Do not identify the new arithmetic as commit `1777a3f` or reuse its approval.

Real-binder builds with Triton 3.9.0, gfx1250/wave32, four waves and
`[513,2048]` give these results. Instruction counts are static ISA instructions,
including scheduling instructions; a dual instruction counts once.

| Path | Frozen native bytes / VGPR / instructions | Previous PS bytes / VGPR / instructions | Selected PS bytes / VGPR / instructions | Selected non-PS bytes / VGPR / instructions |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 1388 / 49 / 205 | 1436 / 53 / 221 | 892 / 21 / 138 | 780 / 14 / 112 |
| QAT | 1264 / 44 / 181 | 1420 / 47 / 209 | 876 / 18 / 131 | 772 / 15 / 114 |

The same selected instruction bodies and resources were built for `[1,32]`
and `[513,2080]` tails. All selected paths have 68 SGPR, shared=0 and scratch=0.
PS VGPRBlocks are 1/1 versus native 3/2; these do not measure occupancy.
Eight general FP32 divide sequences per lane become four packed FP32 multiply
instructions. Both paths now use fewer code bytes and VGPRs than native.
No v_div_* or v_rcp_f32 remains. Static rejection gates are FP8 code >= 1388 B
or VGPR > 49, QAT code >= 1264 B or VGPR > 44, any LDS/spill/scratch,
memory-mask EXEC branch or lost vector width. All selected builds pass.

The group32 layout, grid `(256,)`, four wave32 waves, 1024 elements/task,
`task = pid0 + 256*k`, buffer range and 64-bit tile rebasing remain fixed.
Buffer loads stay 128-bit; FP8 stores stay 64-bit plus a byte scale store;
QAT stores stay 128-bit. No speculative address variant or runtime switch was
added. The installed `amd.cdna3` buffer API lowers to gfx1250 opcodes. The
namespace suggestion N1 remains a nonblocking naming issue outside this patch.

A full FP8 task still reads 2048 B and writes 1024 B plus 32 B of scales:
3104 B/task. QAT reads and writes 2048 B each: 4096 B/task. Wave w writes scales
8w..8w+7 through lanes 0,4,...,28. Each wave issues one scale store with OOB
non-owner offsets. This does not establish a fourfold change in issued stores.
The explicit layout removed the old PS's 32 B dynamic LDS and two TG syncs;
frozen FP8 native has one such sync. QAT already has no LDS-removal gain.
At 96 rows there are only 192 tasks, so no worker amortizes multiple iterations.
The arithmetic reduction is concrete; request combining, 8 B/wave service
cost, cache traffic and runtime performance are unmeasured.

The new ISA was reviewed for global masks, zero filling, single-writer masks,
SCC/VCC, descriptor SGPRs and source VGPRs. FP8 PS preserves the owner predicate
in VCC and uses a separate scalar mask for code 256. Non-PS consumes ownership
into its scale offset before VCC is reused. QAT reuses VCC only after load zero
filling. SCC remains live from the loop comparison to its backedge. Descriptor
rewrites have XACK waits; cross-iteration source reuse follows the protected
PC change. MI400 section 6.9.7.2 requires translation or completion before
source reuse; VM_VSRC alone is insufficient. Tasks do not consume preceding
outputs, so no unconditional per-iteration store-completion wait is added.
No wait or barrier was manually removed. RETRY_DISABLE is unknown;
MODE.REPLAY_MODE does not establish runtime retry state. Private artifacts
retain the line-by-line review of this build, not the previous line numbers.

## 3. Are correctness and review complete?

- [x] Implementation and Codex code/algebra review, including special values,
  default NaN behavior, denorm modes and the final FP8/QAT conversion order.
- [x] CPU exhaustive check: 10,813,440 pairs; final bytes match. No GPU ran.
- [x] Worker local builds: 12 native/candidate AST variants and all 12 frozen
  symbol hashes; 22 real-binder builds include 12 selected paths and tails,
  six native references and four previous-code comparisons. QAT uses `S=None`.
- [x] Collection-only harness: exactly the same 45 IDs. Original assertions,
  finite oracles, five-iteration/tail/ownership/API tests and both A09 Graph
  nodes remain. The Graph nodes retain the three-output tuple and separate
  FP8/scale and QAT input-change assertions for 20 replays.
- [x] Black per-file checks, Ruff, protected-file hashes and diff whitespace.
- [x] Previous findings A1/A2/A3: new dependency review, quantitative static
  gates on both paths, and real `S=None` builds. N2's Graph tuple fix remains.
- [x] A4/A5 and A09-R1/R2/R3: mandatory FP8/QAT timing, explicit row decisions,
  numerical tests without profiling, then two Graph traces with explicit
  profiler branches. The guide states that benchmark defaults exclude A03.
- [ ] Outer runner: execute its unchanged fixed acceptance command and record
  the result. Worker checks do not claim that runner result.
- [ ] Claude independent review of this arithmetic patch. Previous approval
  is historical; any new approval certifies only bounded local readiness.
- [ ] GPU: all 45 tests pass, zero skips, byte/guard mismatches or GPU errors.
  `[513,2048]` has 1026 tasks; two workers run five iterations.
  `[513,2080]` has 1043 tasks; its last task has one valid group.
- [ ] GPU: save actual wrapper JIT and confirm 256 TGs x 128 threads for PS.
- [ ] GPU: pass native-versus-PS performance acceptance for both FP8 and QAT.

A03 `7108dbec079595026f76b7b1b8e082fe1669796d`, its wrapper, A09's wrapper,
benchmark, compiler, frozen native/manifest and submodule declarations remain
unchanged. Public base and frozen-native identities are in the validation guide.
The [historical hardware report](mono-results/2026-11-18-gfx1250/README.md)
belongs to `d5930aff35d67fa0a2e74ba425909747f3ccef9f`: 36 tests passed,
eight timing rows regressed, four were inconclusive. It cannot validate this
candidate or its QAT changes.

## 4. What is the one hardware handoff?

Follow [the validation guide](mono-persistent-validation.md) after the outer
local acceptance and Claude review. Use the supplied final source and existing
ROCm environment. Run all 45 numerical tests without profiling. Then trace
only `test_graph_replay_changes_inputs[a09]` and `[a09-five-iterations]`.
A failed or unavailable profiler leaves launch acceptance pending, preserves
numerical results and stops timing. Save actual JIT and check P256 first.

Then compare only frozen native and selected PS. The default benchmark includes
A09 FP8/QAT; this handoff also selects them explicitly:

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --ops a09 a09-qat --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json"
```

Retain RotatingBufferBench, the common input pool, alternating provider order
and each round's latency, delta and timestamps. Hot/rotating describes input
reuse. Outputs are allocated by wrappers during graph capture.

```text
delta = PS median - native median
noise = native P90 - native P10
regression = delta > noise and PS P10 > native P90
improvement = -delta > noise and PS P90 < native P10
otherwise = inconclusive
```

All six rows for each path must be improvements, with correctness and P256
passing, before that path passes. A regression fails; equal medians, overlap
or insufficient separation remain inconclusive and do not prove equivalence.
There is no positive slowdown allowance. No automatic rerun or tuning sweep
follows a failure or inconclusive result. Return the evidence and separate
FP8/QAT statuses. A gain against old PS cannot offset a native regression.
