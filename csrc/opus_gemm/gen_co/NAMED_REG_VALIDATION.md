# OPUS BF16 CO named-register migration validation

> Historical validation of the earlier std-branch request. Current CO files
> were subsequently rebuilt with amdgpu-pin-hybrid; see
> [HYBRID_VALIDATION.md](HYBRID_VALIDATION.md) for their compiler and target identities.

## Status

The compute (64), wave-layout (155), and ring/pin (2) families have been rebuilt
using the new automatic-local `amdgpu_pin_vgpr` attribute. All **221** candidates
pass actual WMMA destination placement, 64-byte kernarg, LDS, zero-spill, IR tuple
coverage, file SHA256, and numerical checks on d07-2 (`gfx1250`, 256 CUs).

The production CO images are deliberately unchanged. New candidates are in the
separate root below and require an explicit runtime selection:

```sh
# From the aiter repository root, before importing aiter:
export OPUS_GEN_CO_DIR="$PWD/csrc/opus_gemm/gen_co/named_reg"
```

There is no kid, symbol, dtype, host-launcher, cluster, or kernarg ABI change.
There is no new bias, FP32-output, or genuine split-K capability.

## The compiler was rebuilt on d07-2

Source: [requested LLVM branch](https://github.com/yuyzhang512/llvm-project/commits/amdgpu-named-reg-pin-std/).

```text
LLVM HEAD: 583640bc544035fcc87177fee0410e0d87693919
Host compiler: /home/blyu/opus-named-reg-20261026/llvm-build/bin/clang
Container compiler: /wksp/opus-named-reg-20261026/llvm-build/bin/clang
Compiler SHA256: babbb64b088b48b061058a6f92a0d596c169c2cd55cd0be1887fc3bb52082b11
Build: Release, assertions ON, clang/lld, AMDGPU backend
Test container: blyu_opus_named_reg_20261026
Aiter baseline: 745c9cba75e829a9fe7582a3529145a0e10f8af9
```

The new compiler was built from source, not taken from the existing soft-hint
checkout. Its official CodeGenHIP pin-attribute fixture passed FileCheck, and its
SemaHIP pin-attribute fixture passed `-verify`. The emitted candidate IR also
identifies the requested LLVM SHA. No LLVM source patch beyond that branch was
needed for the final kernels.

This minimal AMDGPU-only build reports an unknown default host triple. The real
HIP compilations therefore explicitly use `--target=x86_64-unknown-linux-gnu`;
this is recorded in every candidate's device flags. The reproducible build
configuration can instead set `LLVM_DEFAULT_TARGET_TRIPLE` to that host triple.

## Why the first version regressed, and the final fix

A storage-free bank whose getter re-read a named physical tuple was numerically
correct but wrong for the intended producer placement. Each loop block became
`read_register(C) -> WMMA -> write_register(C)`. LLVM's per-basic-block
`scanNamedRegs` treats read-before-write as a whole-tuple reservation. The WMMA
`tied vdst=src2` update then creates a complex virtual interval that cannot be
coalesced into that reserved physical register. The ISA had high-C/low-C copies
around low-destination WMMAs. The first 14-case ABBA sweep was 1.81–3.80x slower.

The final bank keeps ordinary SSA members between accesses. Each setter declares
an **automatic local with the new attribute**, then immediately copies its value
out to the SSA member:

```cpp
__attribute__((amdgpu_pin_vgpr(Base))) V slot = value;
head = slot;
```

The getter returns the SSA member rather than re-reading the physical register.
This avoids the problematic read-before-write shape and restores direct
accumulator-bank WMMA destinations. The member itself is not falsely advertised
as an attributed, whole-lifetime physical variable: actual destination placement
is a fatal ELF disassembly check in the builder, in addition to IR evidence.

All 221 candidates passed this stronger placement check. The unsafe experimental
wlr combination `OPUS_PIN_A=1` plus `OPUS_C_BF16F32=1` is rejected at compile time:
its A and narrow-C banks have the same base and cannot be real physical aliases
while both remain live.

## Numerical and integration verification

Run baseline and candidate suites in separate processes because the launcher
caches a loaded symbol per process. Both runs reported:

```text
396 passed, 14 skipped, 127 subtests passed
```

The skipped tests require other GPU architectures. Coverage includes all 221
CO kids with partial clusters and simultaneous M/N/K tails, exact routing and
missing-image override behavior, plus 96 added BMM cases spanning tiny K,
one-step and multi-ring tails, independent A/B batches, source row padding,
split spellings 0/1, NaN-poisoned outputs, and repeated graph replays. Numerical
checks use `torch.testing.assert_close(rtol=0.03, atol=0.5)`, not a CLI that merely
logs mismatch ratios.

```sh
# From the aiter root, with a ROCm PyTorch/pytest environment:
OPUS_TEST_CO_ROOT="$PWD/csrc/opus_gemm/gen_co/named_reg" \
python3 -m pytest -q \
  op_tests/test_opus_co_builder.py \
  op_tests/test_opus_co_integration.py \
  op_tests/test_opus_a16w16_registry.py \
  op_tests/test_opus_a16w16_policy_parity.py \
  op_tests/test_opus_a16w16_gemm.py

# Baseline: a fresh process without OPUS_TEST_CO_ROOT.
```

## Final performance

Same d07-2 GPU, seed, tensors, shapes, kids, and hot-cache HIP graph method.
ABBA process order: baseline, named, named, baseline. Each process uses 50 kernels
per graph, 20 graph replays per sample, and 7 samples after warmup. The table pools
14 samples for each implementation. Clocks were not forcibly fixed; small
percent-level differences are not evidence of a speedup.

| kid | M/N/K | production us | named us | named/production |
|---:|---|---:|---:|---:|
| 21063 | 4096/2048/7168 | 59.619 | 60.103 | 1.008x |
| 21063 | 512/2048/4096 | 20.438 | 20.133 | 0.985x |
| 21105 | 4096/2048/7168 | 84.081 | 84.068 | 1.000x |
| 21105 | 512/2048/4096 | 13.032 | 12.935 | 0.993x |
| 21177 | 4096/2048/7168 | 60.163 | 60.158 | 1.000x |
| 21177 | 512/2048/4096 | 19.125 | 19.294 | 1.009x |
| 21208 | 4096/2048/7168 | 92.360 | 92.696 | 1.004x |
| 21208 | 512/2048/4096 | 11.571 | 11.689 | 1.010x |
| 21316 | 4096/2048/7168 | 75.880 | 75.765 | 0.998x |
| 21316 | 512/2048/4096 | 35.253 | 35.177 | 0.998x |
| 21317 | 4096/2048/7168 | 75.049 | 75.168 | 1.002x |
| 21317 | 512/2048/4096 | 35.217 | 35.345 | 1.004x |
| 21301 | 4096/2048/7168 | 132.179 | 132.156 | 1.000x |
| 21301 | 512/2048/4096 | 16.099 | 16.404 | 1.019x |

Median latency ratio: **1.001x**; range **0.985x–1.019x**. The large first-version
regression is resolved; no material overall speedup is claimed. This is a
representative 14-case comparison, not a full-model retuning or all-shape
performance certification.

For all seven measured kids, the final VGPR counts match the production counts.
For example, 21063 has 768/768 actual accumulator-bank WMMA destinations and
21 VA_VDST(0) waits; 21105 has 384/384 and 19 waits; 21316 has 1024/1024 and 38
waits. Historical wlr `wmma_dst_hi` metadata counted only v256–v511; the new
counter correctly includes the entire v256–v1023 addressable high register file,
so the old 512/1024 metadata must not be mistaken for half of its actual WMMAs
being low.

## Rebuild on the existing test environment

```sh
ssh mi450-d07-2
# In container blyu_opus_named_reg_20261026:
cd /wksp/opus-named-reg-20261026/candidate
python3 csrc/opus_gemm/gen_co/build_co.py \
  --llvm-bin /wksp/opus-named-reg-20261026/llvm-build/bin \
  --hipcc /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/hipcc \
  --device-flag=--target=x86_64-unknown-linux-gnu \
  --out-dir csrc/opus_gemm/gen_co/named_reg/gfx1250 \
  --jobs=24 --fail-on-spill
```

The original remote user worktrees and the other user's container were not
modified. The isolated compiler, candidate checkout, per-kernel IR/ISA listings,
raw ABBA JSON data, and test logs remain in the task-owned test directory for
reproduction. No tuned CSV was modified and no production CO was replaced.
