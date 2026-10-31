# Round F: fp8-only Q LDS staging on champion C

Base: `177419ad4c6e56fbbbaabbbb5eec0fcb0c4ccec6` (kernel SHA256
`90e9de977c185a6ee98068e7de12400ee8194055877fcb7a6f921f5176a52ac0`).
Champion C's CPU gate additions are imported from its gates commit `da144769`.

## Change and scope

Only `_build_kernel` changes in the production kernel module. Build-time Q
storage/load/barrier closures select LDS for fp8 and the existing global Q
vector loads for the bf16 scalar fallback. The separate WMMA builder, wrapper,
E4M3 software decoder, global access helpers, descriptors and safety guards are
unchanged. Historical module-header descriptions were intentionally not edited.

For fp8, one static `SharedAllocator(base_alignment=16, static=True)` allocates
256 packed i32 words (1024 bytes). The cooperative loop stages the query row
with one clamped in-bounds scalar load and one predicated/disjoint LDS dword
store per lane at the production 256-thread block size. All lanes, including
output tails and invalid metadata, reach one unconditional workgroup barrier.
The arithmetic loop reads 64 aligned 16-byte chunks from LDS instead of global
Q; Q unpacking and fp8 accumulation/scale/mask/store order are unchanged.

For bf16 scalar, the staging loop has a constexpr zero trip count, Q storage
is a descriptor passthrough, and the barrier helper is a no-op. No LDS allocator
or barrier is selected for bf16. No dev1004 APIs, sentinel stores, or changed
host dispatch API are introduced.

## CPU verification

Run from this directory with CPU PyTorch and `PYTHONDONTWRITEBYTECODE=1`.
Set `QSA_GOLDEN_DIR` to the frozen golden_baseline directory; no frozen files
may be written. No GPU is used by these commands:

```text
python test_contract_cpu.py
python trace_check_cpu.py
python audit_addressing_cpu.py
python test_q_lds_cpu.py
```

The contract gate includes C's WMMA lane emulator and wrapper guard checks.
The real-source symbolic WMMA trace must remain PASS 16/16. The address audit
retains C's WMMA coverage and adds the fp8 global stage/LDS accesses. The Q LDS
gate checks all 65536 bf16 bit patterns, multiple block sizes, unique dword
writers, aligned consumer coverage, and build-time fp8-only source structure.

Stock FlyDSL 0.3.3.dev886 tracing must be run in a no-device container:

```text
COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 python3 compile_only.py
```

All three specializations (bf16 scalar, fp8 LDS-Q, bf16_wmma) must report
`[COMPILED]`. On dev886 this is tracing only, not LLVM/backend compilation or
GPU execution; code_objects is expected to be empty.

## Source preservation evidence

An exact source comparison against the base kernel established:

- Every byte outside `_build_kernel` is unchanged.
- The complete WMMA/comments/wrapper suffix has identical SHA256
  `156b1dee381294a48a3fcd4f15d01766fe585e6e4d93520b2a7030db1f4b3062`.
- Scalar descriptors, metadata/masks, and launch code are byte-identical.
- The entire scalar score/unpack/scale/arithmetic/store block is identical
  after substituting only the Q load seam and relocating `q_base` to staging.

## Recorded results (2026-10-31)

- Windows CPU PyTorch: extended `test_contract_cpu.py` PASS, including C's WMMA
  emulator, scale/decoder cases and all wrapper guards; no skipped guards.
- Real-source `trace_check_cpu.py`: PASS 16/16 on Windows CPU.
- Full `audit_addressing_cpu.py`: PASS 26/26 (19 bf16, 7 fp8), with zero
  out-of-bounds/int32-overflow findings, zero misaligned fp8 LDS accesses, and
  every retained C WMMA gate passing.
- `test_q_lds_cpu.py`: PASS 31/31 on Windows CPU, including every bf16 encoding
  in both halves of packed words and seven block sizes.
- No-device a07-3 container, image
  `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`: compile-only trace PASS 3/3
  (bf16, fp8, bf16_wmma), `compiled_function=NoneType`, `code_objects=[]`.
  Distribution metadata independently verified `flydsl==0.3.3.dev886`.
- Traced kernel SHA256:
  `7ce0acfc2dd9458ce0f065ceb359da2cbd921a01e4b0e9c8d0070e5b8d48e13c`.
- Remote trace evidence:
  `/data/blyu/qsa/results/g2/opt-f-cpu-trace-20261031-1858/trace.log` and
  `compile_only.json`; isolated inputs are under
  `/data/blyu/qsa/work/opt-f-cpu-trace-20261031-1858`.
- C audit preservation comparison: `_wm_lane_addrs` and `audit_wmma` source
  strings match C exactly; all original analytic/enumerated result fields match
  for its 19 bf16 cases, and boundary WMMA result dictionaries are unchanged.

## Expected benefit and GPU handoff

No GPU measurements were made by the optimizer. Based only on champion B's
previously measured fp8 LDS-Q change, decode_fp8_c65536 is expected to recover
from about 413.8 us to about 360.6 us: approximately 12.9% lower latency (14.8%
throughput speedup), not a measured claim for this port. bf16 WMMA is expected
to retain C's performance because its code is unchanged. Other fp8 improvements
are unquantified until root measures.

Root must perform a stock-dev886 real build and the established GPU correctness
suite under the cooperative GPU lock, including tails, invalid metadata/pages,
fp32 scale cases, and masked -inf outputs. Compare fp8 outputs bitwise with C
where practical; unchanged arithmetic plus exhaustive Q-bit preservation is not
a substitute for GPU equivalence. Then run unprofiled A/A, A/B and held-out
reruns against C and B on the same GPU/environment. Check LDS lowering,
barriers, occupancy and spills if expected gains are not recovered. Never
upgrade FlyDSL and never push this branch.
