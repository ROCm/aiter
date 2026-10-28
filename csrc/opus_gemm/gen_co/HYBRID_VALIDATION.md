# Hybrid LLVM gfx1250 / A0 strict CO rebuild

## Compiler and artifact identity

All registered OPUS BF16 CO variants were rebuilt from the requested
[amdgpu-pin-hybrid branch](https://github.com/yuyzhang512/llvm-project/commits/amdgpu-pin-hybrid/).

```text
LLVM commit: 352c37bad3561fc0f23163d5b1419910213d26bb
Compiler: clang 24.0.0git, Release + assertions
Compiler SHA256: 95b17b6bc5a2ae222ed5024113c3b4891c5f84989dfe92fa602b1f610d3922a8
Build machine: g21-10
Private build: /home/blyu/opus-hybrid-co-20261026/llvm-build
```

The compiler was built from source. Its official pin-attribute CodeGenHIP
fixture passed FileCheck. ClangIR was disabled, and the sparse source checkout
includes libc/libunwind common headers needed by this LLVM/LLD revision.

| profile | entries | compiler target | actual ELF processor | runtime root |
|---|---:|---|---|---|
| regular | 221 | `gfx1250` | `0x49` | `gen_co/` |
| A0 | 221 | `gfx1250-strict` | `0xeb` | `gen_co/gfx1250-a0/` |

The regular files are also mirrored in `gen_co/named_reg/` to retain the previous
explicit-root interface. Regular images use `gfx1250/`; A0 images use the flat
sibling `gfx1250-a0/`. The logical runtime architecture remains gfx1250. Kids, filenames, exported symbols,
launch bounds, cluster dimensions, and the 64-byte argument structure are stable.

Each profile contains **87,968 WMMA instructions**, all with actual destinations
inside the requested accumulator/output register banks. Every file passed:

- named-register read/write coverage for all expected tuples;
- actual ELF disassembly producer-destination validation;
- bare ELF64 little-endian identity and distinct processor flag;
- exactly one expected kernel symbol, 64-byte kernarg, and expected LDS size;
- zero VGPR/SGPR spills and no unresolved device symbols;
- SHA256 validation after archive transfer and installation.

Archive hashes:

```text
regular: e9644121eb1b9f8a4e41b3b80a3d536813e994e804e6ab8cabe75f573021c888
a0:      73709c79a3c738cec369b2cfa8893308efe20ab7c2af439a34c2a4ecafa66776
```

## Required compiler compatibility adjustments

The hybrid revision changes the TDM builtin descriptor groups to unsigned
4/8-lane vectors. Earlier clang-24 changed only group0, while clang-23 used signed
groups. The shared OPUS wrapper now uses C++17 SFINAE to detect the descriptor
prototype and equal-width bit casts; it does not alter descriptor bits, order,
gather indices, or high-dimensional fields. A C++17 load/store instantiation
probe passed on the new strict compiler, in addition to the C++20 CO builds.

Clang defines `__gfx1250_strict__` for the distinct strict CPU. The offline stub
aliases the shared source-layout guard to `__gfx1250__` before including OPUS,
but retains the actual strict compiler target, target ID, bundle ID and ELF flag.

The strict BF16 kernels are standalone and use `-nogpulib`: the installed SDK
has no strict ISA device bitcode, and these kernels require no OCML/OCKL calls.
The builder checks the final ELF for unresolved symbols rather than masking an
external dependency. It never relabels a strict ELF as regular.

## Runtime verification

Regular profile: on the previously authorized d07-2 B0 machine, the complete
CO integration suite passed **332 tests**. This includes every registered CO kid,
partial-cluster and M/N/K tails, BMM/source-padding cases, and repeated graph
replays with strong numerical assertions.

A0 profile: on c7-1, ROCm 10.0 / HIP `7.15.26333` rejected the real strict ELF with
`hipErrorNoBinaryForGpu`. The machine's already-available ROCm 10.1 / HIP
`7.16.26385` runtime successfully loaded it through both the HIP driver and the
existing aiter path. Three family representatives (21016, 21105, 21316) passed
CPU-golden BMM checks at tiny `1x1x1` and ragged `129x129x385` shapes.

The exhaustive A0 suite is **not certified**. The GPU-input suite reported
repeated failures; an isolated diagnostic then stalled in GPU random-input setup
before any CO call. A separate CPU-input diagnostic later stalled in the host-to-
device copy, also before a CO call. On g21-10, another verification attempt
blocked in its `rocminfo` child; a known-architecture retry also produced no
forward progress. Only this task's private test/query processes were stopped;
no shared GPU reset or other-user workload interruption was performed.
Two task-owned g21-10 `rocminfo` read queries (host PIDs 3030570 and 3059345)
remain in uninterruptible `D` state despite termination requests. They are inside
`blyu_hybrid_co_g21_10_20261026`; driver/device recovery is needed to release those
waits. No pytest or kernel-execution worker remains in that private container.

The tests now offer explicit `OPUS_CO_CPU_REFERENCE=1` to generate data and an
independent strong golden on CPU while retaining all selected GPU CO, tail,
BMM/padded-stride and graph-replay checks. This mode is not a claim that the
blocked exhaustive run passed. Do not interpret successful compilation or the
six CPU-golden smoke cases as full all-shape A0 correctness certification. A
healthy available A0/runtime test window is still required for that final step.

No performance improvement is claimed by this rebuild. No other user's
container, GPU clocks, firmware, or workload was modified. All validation uses
private containers and directories.

## Local kid namespace after the registry migration

The current manifest and all three build-info tables use local kids 496–716 in
the `a16w16/gfx1250/CU` namespace. The 221 records were matched one-to-one by full
kernel configuration/symbol; the old global IDs cannot be converted by a single
offset. Representative mappings are 21016→496, 21063→543, 21105→544,
21177→596, 21208→612, 21316→715 and 21317→716. CO bytes, filenames and exported
symbols are unchanged. IDs quoted in the earlier validation logs below refer to
the historical global namespace used when those runs were performed.

## Requested ROCm dev-image launch verification

A subsequent native-HIP test was performed on g21-10 in the exact
`rocm/dev-ubuntu-24.04:10.1.0-full` image (HIP `7.16.26385`). The initial run
successfully loaded and launched the current ASIC-0 strict COs for kids 21016,
21105 and 21316: six tiny/tail, batch-2 cases matched the CPU golden exactly.
No PyTorch, GPU RNG or rocminfo was used. A repeat full-list run later stopped
progressing on the 21316 tail case, so it does not certify all 221 variants.
Details and exact reproduction commands are in
[A0_DEVFULL_LAUNCH.md](A0_DEVFULL_LAUNCH.md).

## Automatic ASIC Revision selection

The host CO loader now reads the current HIP device's `hipDeviceProp_t.asicRevision`,
the same HSA ASIC Revision field printed by `rocminfo`: `0x0` selects the A0
`gfx1250-a0/` subtree and `0x1` (or later) selects regular `gfx1250/`. It does not
read PCI/chip revision, take the minimum across devices, or spawn `rocminfo`.
Default imports export the combined root, so manual A0/B0 environment switching
is no longer needed and the exact kid/symbol remains the same.

Cache keys include the current HIP device and resolved absolute CO path. Generated
launchers no longer retain their first kernel in a static reference. A direct
leaf override is supported without adding another subdirectory, but wrong/missing
profiles fail before registration. Actual ELF processor identity is checked
against the detected ASIC; A0 never falls back to regular images. Registration
failure now rolls back its fatbinary before the image buffer is destroyed.

Host validation completed: 5 Python ASIC/parser/codegen/path tests plus 12 builder
regressions; compiled C++17 path/profile tests; and a compiled mock-HIP test of
the actual loader/cache/registration code covering device 0→1→0, changed roots,
direct A0 roots, concurrent threads, wrong-profile rejection, and registration-
failure cleanup/retry. These tests do not certify the remaining real A0 numerical
coverage described above.

Environment changes must not race launches. Same-path file replacements, symlink
retargeting and device/context resets require a fresh process. Existing captured
graphs retain their old function; modules are not evicted during graph/launch use.
Callers must keep tensor device, current HIP device and stream consistent; the
CO entry verifies the first two and normal Python entry points supply the stream.

## Build and select

```sh
# From the aiter root, choose the same rebuilt hybrid compiler:
python3 csrc/opus_gemm/gen_co/build_co.py --llvm-bin /path/to/hybrid/bin \
  --offload-arch=gfx1250 --out-dir csrc/opus_gemm/gen_co/gfx1250 \
  --jobs=24 --fail-on-spill
python3 csrc/opus_gemm/gen_co/build_co.py --llvm-bin /path/to/hybrid/bin \
  --offload-arch=gfx1250-strict --out-dir csrc/opus_gemm/gen_co/gfx1250-a0 \
  --jobs=24 --fail-on-spill

# Rebuild the host launcher for the automatic selector; device COs are unchanged.
# Remove an old manual direct-profile override to use the packaged combined root.
unset OPUS_GEN_CO_DIR
AITER_REBUILD=1 python3 -m pytest -q \
  op_tests/test_opus_co_integration.py -x --tb=short

# Optional explicit combined root; current-device ASIC still selects the profile:
export OPUS_GEN_CO_DIR="$PWD/csrc/opus_gemm/gen_co"
```

The builder's `--arch` names the logical runtime directory; `--offload-arch`
names the actual CPU. Compile, unbundle, decode, and ELF validation all use the
actual target. Current registrations are isolated by device and resolved image
path, so root changes re-resolve the same kid. A captured graph remains bound to
the function it captured; same-path file replacements require a fresh process.
