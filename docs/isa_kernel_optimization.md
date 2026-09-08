# ISA-Level Kernel Optimization with LLVM Tools

How to inspect, extract, modify, reassemble, test and profile the pre-built
assembly kernels that AITER ships as code objects (`hsa/<arch>/<family>/*.co`),
using only the LLVM tools bundled with ROCm.

> **Scripts and Dockerfile:** [`docs/examples/isa_optimization/`](examples/isa_optimization/).
> Everything below can be reproduced with `bash docs/examples/isa_optimization/roundtrip.sh <kernel.co>`.

## Overview

AITER's fastest paged-attention, MLA, fMHA, fMoE, GEMM and top-k kernels are
hand-written assembly, shipped as ELF code objects and loaded at runtime with
`hipModuleLoad`.  Their source is not in the repository, but the code objects
carry enough information to rebuild an equivalent, editable assembly file:

| In the `.co`                        | Contains                                                               | View with                                   |
|-------------------------------------|------------------------------------------------------------------------|---------------------------------------------|
| `.text`                             | the instructions                                                       | `llvm-objdump -d --mcpu=<gfx>`              |
| `.symtab`                           | kernel entry symbol **and the original branch labels** (`label_XXXX`)  | `llvm-readelf -s`                           |
| `.rodata` (`<kernel>.kd`, 64 bytes) | the *kernel descriptor*: VGPR/SGPR/AGPR allocation, LDS size, enabled system registers, FP modes | `llvm-objdump -D -j .rodata --mcpu=<gfx>` |
| `.note` (`NT_AMDGPU_METADATA`)      | msgpack metadata: kernel-argument names/offsets/sizes, register counts, workgroup size | `llvm-readelf --notes`               |

The workflow is:

1. **Extract** a standalone `.s` from the `.co` (instructions + kernel descriptor + metadata).
2. **Reassemble** it with `clang` and **verify** the result is byte-identical to the original.
3. **Edit** the `.s` — instructions *and*, when needed, the resource directives.
4. **Test** the rebuilt `.co` with the existing op tests (drop it into the `hsa/` tree).
5. **Profile** with `rocprofv3` (dispatch timing, counters, thread trace).

What you do *not* get back: comments, macros, register aliases or any structure
above the instruction level; the kernels were also not assembled with LLVM, so
a few encodings do not survive the round trip bit-for-bit (listed below).
Budget accordingly — a paged-attention kernel is ~3,000 instructions, an MLA
decode kernel ~5,500.

## Prerequisites

- ROCm with its LLVM tools in `$ROCM_PATH/llvm/bin` (default `/opt/rocm/llvm/bin`):
  `llvm-objdump`, `llvm-readelf`/`llvm-readobj`, `llvm-objcopy`, `clang`, `ld.lld`.
  The LLVM must know your target; check with
  `clang --target=amdgcn-amd-amdhsa --print-supported-cpus 2>&1 | grep gfx950`.
  (Verified with the LLVM in ROCm 6.x/7.x and with upstream LLVM 18 and newer.)
- Python 3.8+ for the helper scripts.
- A GPU is only needed for steps 4–5.
- The ISA reference for your target, for instruction semantics, latencies and —
  most importantly — the data-hazard rules:
  [AMD Instinct MI300 / CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf),
  [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
  (all generations: [GPUOpen ISA documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)),
  plus the [LLVM AMDGPU usage guide](https://llvm.org/docs/AMDGPUUsage.html) for the
  assembler directives, kernel descriptor layout and code-object format.

## Step 1: Locate the kernel and understand how AITER uses it

### Files

```
hsa/
├── codegen.py                 # turns the per-family CSVs into a C++ dispatch table at build time
├── gfx942/
│   ├── pa/
│   │   ├── pa_asm.csv         # (dtype, kv dtype, GQA, mtp, ...) -> kernel symbol + .co file
│   │   ├── pa_bf16_pertokenFp8_gqa16_2tg_4w.co
│   │   └── ...
│   ├── mla/  fmha_v3_fwd/  fmoe/  bf16gemm/  topksoftmax/  ...
└── gfx950/
    └── ...
```

The examples below use `hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co`
(paged attention decode, bf16 queries, per-token FP8 KV cache, GQA ratio 16).

### How a `.co` is selected and loaded

- The host side of each family lives in `csrc/py_itfs_cu/asm_<family>.cu`
  (`asm_pa.cu` here).  It picks a row of the CSV via a heuristic
  (`get_heuristic_kernel`) and constructs `AiterAsmKernel(knl_name, co_name)`.
- `AiterAsmKernel` (`csrc/include/aiter_hip_common.h`) reads
  `$AITER_ASM_DIR/<arch>/<family>/<co_name>` and calls `hipModuleLoad` /
  `hipModuleGetFunction(knl_name)`.  Each load is logged at INFO level:
  `[aiter] LoadKernel: <knl_name> hsaco: <full path>` (`AITER_LOG_LEVEL`
  controls verbosity).  Modules are cached per process, so restart the process
  after swapping a file.
- `import aiter` **sets** `AITER_ASM_DIR` to `<AITER_META_DIR>/hsa/` (see
  `aiter/jit/core.py`), overriding anything exported in the shell.  For a
  `python3 setup.py develop` install that directory is the repository's own
  `hsa/` tree, so the simplest way to run a modified kernel is to replace the
  file there (git keeps the original: `git checkout -- hsa/...` restores it).
  From your own Python code you can instead point `os.environ["AITER_ASM_DIR"]`
  at a modified copy of the tree after importing `aiter` and before the first
  kernel launch.
- The CSV is compiled into the extension when the module is (JIT-)built:
  replacing a `.co` under the same file/symbol name needs no rebuild, adding a
  new row does.
- The loader validates the LDS size declared in the metadata against the device
  limit (`validate_hsaco_lds`), so keep the metadata truthful when you change it.

### Kernel arguments and launch geometry

The kernel-argument buffer layout is defined twice and both are worth having
open while reading the disassembly:

- the packed `KernelArgs` struct in `csrc/py_itfs_cu/asm_pa.cu`
  (`ptr_O`, `ptr_Q`, `ptr_K`, `ptr_V`, `ptr_BT` block tables, `ptr_CL` context
  lengths, `ptr_KQ`/`ptr_VQ` KV scales, `sclg`, `mblk`, `batch`, `Qs`, `Bs`,
  `KVs`, `mtp`, `GQA`, ...), together with the launch in the same file
  (grid = `num_kv_heads x batch x ...`, block = 256 threads = 4 waves);
- the `.args` list in the code object's metadata (`llvm-readelf --notes`), which
  carries the same names with byte offsets:

```
$ llvm-readelf --notes pa_bf16_pertokenFp8_gqa16_2tg_4w.co
amdhsa.kernels:
  - .args:
      - .name: O      .offset: 0     .size: 8   .value_kind: global_buffer
      - .name: Q      .offset: 16    .size: 8   .value_kind: global_buffer
      - .name: K      .offset: 32    ...
      - .name: KQ     .offset: 96    ...
      - .name: sclg   .offset: 128   .size: 4   .value_kind: by_value
      - .name: mblk   .offset: 144   ...
      ...
    .group_segment_fixed_size: 32768
    .kernarg_segment_size: 256
    .sgpr_count:     96
    .vgpr_count:     256
    .wavefront_size: 64
    .max_flat_workgroup_size: 256
```

The prologue of the kernel then reads directly: `s[0:1]` holds the kernarg
pointer (the descriptor enables only `user_sgpr_kernarg_segment_ptr`), so
`s_load_dwordx2 s[12:13], s[0:1], 0x10` is `ptr_Q`, `s_load_dword s64, s[0:1], 0x80`
is `sclg`, and `s2/s3/s4` are the workgroup ids x/y/z.

## Step 2: Inspect the code object

```bash
CO=hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co
LLVM=${ROCM_PATH:-/opt/rocm}/llvm/bin

$LLVM/llvm-objdump -d --mcpu=gfx942 $CO > kernel.isa      # instructions
$LLVM/llvm-objdump -D -j .rodata --mcpu=gfx942 $CO         # kernel descriptor, decoded
$LLVM/llvm-readelf --notes $CO                             # metadata (YAML)
$LLVM/llvm-readelf -sW $CO                                 # symbols: kernel, <kernel>.kd, label_*

python3 docs/examples/isa_optimization/analyze_kernel.py isa $CO   # all of the above, summarized
```

The architecture is recorded in the ELF header (`llvm-readelf -h`: `Flags: 0x54c,
gfx942, xnack, sramecc`); the scripts read it from there, and `--mcpu` must
match it exactly or newer instructions decode as `.long 0x........`.

### Disassembly format

`llvm-objdump` prints AMDGPU code with the instruction first and the address
and encoding as a trailing comment — the reverse of its x86 layout — precisely
so that the text is (almost) valid assembler input:

```
0000000000002200 <_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE>:
        s_and_b32 s1, s1, 0xffff                                   // 000000002200: 8601FF01 0000FFFF
        s_load_dwordx2 s[8:9], s[0:1], 0x0                         // 000000002208: C0060200 00000000
        ...
        s_cbranch_scc0 label_07A8                                  // 000000002B5C: BF840550

0000000000002b60 <label_0258>:
        s_waitcnt vmcnt(8) lgkmcnt(0)                              // 000000002B60: BF8C0078
```

Branch targets appear by name because the `label_XXXX` symbols are real local
symbols in `.symtab`.  Two caveats the extraction script takes care of:
several labels can alias one address while objdump prints only one of them as
a header (the others still appear as branch operands), and the hexadecimal
suffix is just the generator's naming scheme — do not derive addresses from it.

### Kernel descriptor

```
$ llvm-objdump -D -j .rodata --mcpu=gfx942 $CO
00000000000011c0 <_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE.kd>:
.amdhsa_kernel _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE
        .amdhsa_group_segment_fixed_size 32768      <- static LDS bytes
        .amdhsa_private_segment_fixed_size 0        <- scratch bytes per lane
        .amdhsa_kernarg_size 0
        .amdhsa_accum_offset 256                    <- first AGPR (gfx90a+ unified register file)
        .amdhsa_next_free_vgpr 256                  <- VGPRs (arch + accum)
        .amdhsa_next_free_sgpr 104                  <- see note in Step 3
        .amdhsa_float_denorm_mode_32 0              <- f32 denormals flushed
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_dx10_clamp 0                        <- NB: LLVM's default is 1
        .amdhsa_ieee_mode 0                         <- NB: LLVM's default is 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 0           <- only v0 = workitem id x
        .amdhsa_user_sgpr_kernarg_segment_ptr 1     <- s[0:1] = kernarg pointer
        ...
.end_amdhsa_kernel
```

This block, not the metadata, is what the hardware is programmed from.  With
256 VGPRs per lane a gfx942 SIMD holds two of these waves; that is the number
to move if you are after occupancy.

## Step 3: Extract a standalone `.s`

```bash
python3 docs/examples/isa_optimization/extract_asm.py $CO -o kernel.s
# _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE: 3085 instructions, 6 labels
```

`kernel.s` has three parts:

```asm
        .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
        .text
        .globl  _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE
        .p2align 8
        .type   _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE,@function
_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE:
        s_and_b32 s1, s1, 0xffff
        s_load_dwordx2 s[8:9], s[0:1], 0x0
        ...                                        (1) instructions and labels
        s_endpgm
        .size   _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE, .-_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE

        .rodata
        .p2align 6
.amdhsa_kernel _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE
        .amdhsa_group_segment_fixed_size 32768
        ...                                        (2) kernel descriptor
.end_amdhsa_kernel

        .amdgpu_metadata
---
amdhsa.kernels:
  - .args: ...
    .vgpr_count: 256
    ...                                            (3) metadata, verbatim
amdhsa.version:
  - 1
  - 0
...
        .end_amdgpu_metadata
```

Two adjustments are made to the decoded descriptor so that LLVM's assembler
accepts it and encodes the same 64 bytes:

- `.amdhsa_next_free_sgpr`: the disassembler prints the *granulated total*
  (`(field + 1) * 8`, e.g. 104 or 112), which already includes the six SGPRs the
  assembler reserves implicitly (VCC, FLAT_SCRATCH, XNACK_MASK), together with
  `.amdhsa_reserve_vcc 0`.  Feeding that back fails with `error: value out of
  range` whenever the total exceeds 102.  The script emits `total - 8` (capped
  at 102) with `.amdhsa_reserve_vcc 1`, which produces the identical
  `GRANULATED_WAVEFRONT_SGPR_COUNT`.  When you change SGPR usage yourself, set it
  to the highest SGPR number you use plus one.
- `.amdhsa_reserve_xnack_mask` is dropped: it must agree with the target id, and
  AITER's objects are built for plain `gfx942`/`gfx950` (xnack "any"), for which
  the directive is rejected (`does not match target id`).  Keep the plain target
  id — an object built for `gfx942:xnack-` will not load in a process running
  with XNACK enabled.

`extract_asm.py --text-only` emits part (1) alone, for the in-place patching
flow described at the end.  `--symbol` selects a kernel if a `.co` contains
several (the shipped ones contain one each).

## Step 4: Reassemble and verify

```bash
$LLVM/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -o kernel_rebuilt.co kernel.s
```

Without `-c`, clang runs `ld.lld` and produces a complete code object with the
kernel descriptor relocated and the metadata note attached — `hipModuleLoad`
accepts it directly.  `roundtrip.sh` does the extraction, the build and the
comparison in one go:

```
$ bash docs/examples/isa_optimization/roundtrip.sh $CO
...
=== Step 4: Verify against the original ===
  .text                        identical (19236 bytes)
  .rodata (kernel descriptor)  identical (64 bytes)
  metadata (.note)             identical
  ELF e_flags (target id)      identical (0x54c)

PASS: .../kernel_rebuilt.co is equivalent to the original and can replace it as-is.
```

Establish this baseline for the exact kernel you intend to modify *before*
editing anything; from then on every difference is one you introduced.  The
same check by hand:

```bash
for s in .text .rodata; do
  $LLVM/llvm-objcopy -O binary -j $s $CO orig$s.bin
  $LLVM/llvm-objcopy -O binary -j $s kernel_rebuilt.co new$s.bin
  cmp orig$s.bin new$s.bin && echo "$s identical"
done
diff <($LLVM/llvm-readelf --notes $CO) <($LLVM/llvm-readelf --notes kernel_rebuilt.co)
```

### Known re-encoding differences

All gfx942 kernels checked (paged attention, MLA, bf16 GEMM, top-k softmax) and
all gfx950 paged-attention and MLA kernels round-trip byte-for-byte.  The
gfx950 fMoE kernels that use `v_mfma_scale_f32_16x16x128_f8f6f4` do not: the
shipped encoding has bits 13–14 of the leading `v_mfma_ld_scale_b32` word set
(`op_sel`/`op_sel_hi` of the unused third source, e.g. `D3AC6000`), LLVM's
disassembler ignores them and its assembler emits them cleared (`D3AC0000`).
Everything else in those kernels is identical.  Treat such a rebuild as
functionally unverified until the op test (Step 6) passes on hardware.

When `.text` differs, `roundtrip.sh` prints the first differing instruction in
both versions; by hand, `cmp -l orig.text.bin new.text.bin | head` gives the
byte offset, and `.text` start + offset is the address to look up in the
`// ADDR:` comments of both disassemblies.

## Step 5: Modify

Edit `kernel.s`, rebuild with the `clang` command above, re-run the verification
to see exactly which bytes changed, then test (Step 6).  Because the descriptor
and metadata are part of the source, code size, register allocation and LDS
usage may all change.  Things the assembler will **not** do for you:

### Data hazards and wait states are your responsibility

LLVM inserts hazard `s_nop`s and `s_waitcnt`s when it *compiles* kernels; when
it *assembles* hand-written code it encodes exactly what is written.  The
`s_nop`s in these kernels are almost all mandatory wait states, for example:

```asm
        v_rcp_f32_e32 v50, v50
        s_nop 1                     ; transcendental VALU result read by the next VALU op:
        v_mul_f32_e32 v50, 0x43700000, v50      ; wait states required on gfx940+, nothing checks it
```

Others separate VALU writes of SGPRs/VCC from their readers, DPP operands from
the instruction that produced them, and dependent MFMAs.  An `s_nop` directly
after an MFMA is usually such a dependency distance, not idle time that can be
filled for free — whatever replaces it must be independent of the MFMA *and*
provide at least as many wait states.

Before moving, removing or inserting instructions, check the "manually inserted
wait states" (data dependency) table and the MFMA dependency rules in the ISA
guide for your target: required distances between a VALU/MFMA writing a
register and specific consumers, `s_setreg`/`s_getreg`, LDS-direct and DPP
sources, and the number of independent instructions (or `s_nop` cycles)
required between dependent MFMAs, which depends on the MFMA shape and pass
count.  An `s_nop N` provides N+1 wait states; an unrelated instruction provides
at least one.  Violations do not fault — they silently read stale data.

Likewise every `s_waitcnt vmcnt(..) lgkmcnt(..)` encodes how many memory
operations of each class may still be outstanding at that point.  Reordering
loads, LDS operations or scalar loads changes those counts; recompute them
rather than copying the old values (`vmcnt` counts VMEM/buffer/global ops in
issue order, `lgkmcnt` counts LDS, SMEM and messages; SMEM may return out of
order, so scalar loads are normally waited for with `lgkmcnt(0)`).

### Changing register or LDS usage

- More/fewer VGPRs or AGPRs: update `.amdhsa_next_free_vgpr` (total of both on
  gfx90a/gfx942/gfx950) and `.amdhsa_accum_offset` (first AGPR, multiple of 4),
  plus `.vgpr_count`/`.agpr_count` in the metadata.  Occupancy only changes at
  allocation-granule boundaries (8 registers); the descriptor, not the metadata,
  is what the hardware uses.
- More SGPRs: `.amdhsa_next_free_sgpr` (max 102) and `.sgpr_count`.
- More LDS: `.amdhsa_group_segment_fixed_size` and `.group_segment_fixed_size`
  (64 KiB per workgroup on gfx942, 160 KiB on gfx950; AITER checks the metadata
  value at load time).
- Leave `ieee_mode`, `dx10_clamp` and the denorm modes as shipped unless you
  mean to change numerical behaviour; note they differ from LLVM's defaults, so
  do not delete the directives either.

### Where the time goes

Use the thread trace (Step 7) rather than guessing from the listing: it shows
per-instruction issue and stall cycles for real waves, which makes it obvious
whether a loop is bound by MFMA issue, `s_waitcnt` on memory, LDS bank
conflicts or dependency stalls, and therefore whether rescheduling,
prefetch-distance changes or LDS layout changes are worth attempting.

## Step 6: Test the modified kernel

Run the rebuilt kernel through the existing op tests before measuring anything.
In a develop install the loader reads from the checkout's `hsa/` tree (Step 1),
so build straight over the shipped file and let git keep the original:

```bash
$LLVM/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 \
    -o hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co kernel.s

# run the family's test (see --help for dtype / head-count / context-length selection)
python3 op_tests/test_pa.py
# [aiter] LoadKernel: _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE hsaco: .../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co

git checkout -- hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co     # back to the original
```

The tests compare against a reference implementation with tolerances.  For a
pure rescheduling change the stronger check is bitwise equality with the
original kernel's output on identical inputs: run the same case with the
original and the modified object (same seed) and compare the saved tensors with
`torch.equal`.  Make sure the configuration you test actually dispatches to the
kernel you changed — the `LoadKernel` log line names the file, and
`--kernel-trace` below names the symbol.

## Step 7: Profile with rocprofv3

### Dispatch timing

```bash
rocprofv3 --kernel-trace --stats --kernel-include-regex 'pa_bf16_pertokenFp8_gqa16' \
          -d ./prof_orig -- python3 op_tests/test_pa.py
# swap in the modified .co (Step 6), then
rocprofv3 --kernel-trace --stats --kernel-include-regex 'pa_bf16_pertokenFp8_gqa16' \
          -d ./prof_mod  -- python3 op_tests/test_pa.py
```

`--stats` writes a per-kernel summary (count, total/mean/min/max duration) next
to the trace; `--kernel-include-regex`/`--kernel-exclude-regex` keep the output
to the kernels of interest.  The default output is a `rocpd` SQLite database
(`-f csv`, `json`, `pftrace` for Perfetto, `otf2` are also available);
`analyze_kernel.py profile ./prof_orig --filter pa_` prints per-kernel
statistics and the VGPR/AGPR/SGPR/LDS values the runtime recorded, straight
from the database.  Add `--memory-copy-trace --hip-trace` for an end-to-end
timeline.

### Hardware counters

```bash
rocprofv3 -L                                     # list counters available on this GPU
rocprofv3 --pmc SQ_WAVES SQ_BUSY_CYCLES --kernel-include-regex 'pa_bf16' -d ./pmc -- python3 ...
```

Only a handful of counters can be collected per pass; rocprofv3 re-runs the
application when more are requested.  For derived metrics (occupancy limiters,
LDS bank conflicts, cache hit rates, achieved bandwidth) use
[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/)
which drives the same counters.

### Thread trace (ATT): per-instruction timing

ATT records the instruction stream of the waves on one compute unit with issue
and stall cycles, and maps it back onto the disassembly — the tool of choice
for judging an ISA change.  Decoding requires `librocprof-trace-decoder.so`,
which is not part of the ROCm 7.2 packages: install a release from
<https://github.com/ROCm/rocprof-trace-decoder/releases> into `/opt/rocm/lib`, or
build it from source:

```bash
git clone --depth 1 --branch develop --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
cd rocm-systems && git sparse-checkout set projects/rocprof-trace-decoder
cd projects/rocprof-trace-decoder
cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm
cmake --build build -j$(nproc) && cmake --install build
```

Trace the kernel of interest.  By default only the *first* dispatch of each
matching kernel is traced; pick a warmed-up iteration explicitly:

```bash
rocprofv3 --att --kernel-include-regex 'pa_bf16_pertokenFp8_gqa16' --kernel-iteration-range 5-5 \
          --att-target-cu 1 -d ./att_out -- python3 op_tests/test_pa.py
```

Useful options: `--att-target-cu` (which CU to trace, default 1),
`--att-shader-engine-mask`, `--att-simd-select`, `--att-buffer-size` (raise it
if the tool reports lost data), `--att-perfcounter-ctrl`/`--att-perfcounters`/
`--att-activity` (stream SQ activity counters alongside the trace on gfx9),
`--att-consecutive-kernels`; see `rocprofv3 --help` for the defaults of your
version.  The output directory contains `stats_*.csv` (per-instruction latency
summary — often enough to compare two versions of a loop), a
`ui_output_agent_<a>_dispatch_<d>/` directory to open in
[ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer)
(source/ISA view with per-instruction issue and stall cycles, wave timeline),
the raw `.att` streams and the `.out` code objects they were decoded against.
Compare the same dispatch of the original and the modified kernel.

## Alternative: patching `.text` in place

If you only reorder or replace instructions and the code size does not grow,
you can splice new code into the original object instead of rebuilding it:

```bash
python3 extract_asm.py $CO --text-only -o text.s          # part (1) only
# ... edit text.s ...
$LLVM/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c -o text.o text.s
$LLVM/llvm-objcopy -O binary -j .text text.o text.bin
cp $CO patched.co
$LLVM/llvm-objcopy --update-section .text=text.bin patched.co
```

Limitations: the section cannot grow (`llvm-objcopy: error: cannot fit data of
size N into section '.text' with size M that is part of a segment`) — pad with
`s_nop 0` or delete dead code to compensate, and remember that padding inside a
loop costs issue slots; label addresses move if sizes change before them (the
assembler recomputes branch offsets, but anything you keep must still fit); and
the kernel descriptor and metadata stay exactly as shipped, so register and LDS
usage cannot change.  The standalone rebuild has none of these restrictions and
is the recommended path.

## Architecture notes

| `--mcpu` | Products        | Notes                                                                                   |
|----------|-----------------|-----------------------------------------------------------------------------------------|
| gfx90a   | MI210 / MI250   | CDNA2, wave64, unified VGPR/AGPR file (`accum_offset`), 64 KiB LDS                        |
| gfx942   | MI300A / MI300X | CDNA3; adds FP8 MFMA, transcendental-VALU hazards; 64 KiB LDS; XNACK-capable (MI300A)    |
| gfx950   | MI350X / MI355X | CDNA4; adds `v_mfma_scale_*` / FP4-FP6 MFMA, 160 KiB LDS; needs an LLVM that lists gfx950 |

AITER ships kernels for gfx942 and gfx950.  The `e_flags` of the shipped objects
use the generic target id (xnack/sramecc "any"); keep it that way when
rebuilding.

## Troubleshooting

**`error: value out of range` on `.amdhsa_next_free_sgpr`** — the value is the
granulated total printed by the disassembler; use the number of SGPRs the code
actually addresses (≤ 102), see Step 3.

**`error: .amdhsa_reserve_xnack_mask does not match target id`** — delete the
directive (generic target id) or make the target id explicit
(`...--gfx942:xnack-` in `.amdgcn_target` and `-mcpu=gfx942 -mno-xnack`), consistently.

**`error: undefined label 'label_XXXX'`** — a branch refers to a label alias that
objdump did not print as a header; take labels from `llvm-readelf -s` (the
script does) rather than from the objdump headers.

**Instructions shown as `.long 0x...`, or "Total instructions: 0"** — wrong or
missing `--mcpu`, or an LLVM that predates the target.

**`hipModuleLoad` fails / "no kernel image is available"** — the object was
built with `-c` (relocatable, no program headers), for a different `--mcpu`, or
with a specific xnack/sramecc setting the runtime does not match; compare
`llvm-readelf -h` `Flags:` with the original.

**`llvm-objcopy: cannot fit data of size ... that is part of a segment`** — the
in-place flow cannot grow `.text`; use the standalone rebuild.

**Rebuilt `.text` differs before you changed anything** — see "Known
re-encoding differences"; locate the instruction with `roundtrip.sh` and decide
from the ISA guide whether the differing bits are meaningful.

**Results wrong after an edit, no fault raised** — almost always a violated
data hazard or a stale `s_waitcnt`; re-read Step 5 with the ISA guide's
dependency tables at hand and diff against the last known-good `.s`.

## References

- [LLVM AMDGPU backend usage](https://llvm.org/docs/AMDGPUUsage.html) — code object format, kernel descriptor fields, `.amdhsa_*` directives, metadata schema
- [AMDGPU instruction syntax](https://llvm.org/docs/AMDGPUInstructionSyntax.html) and [operand syntax](https://llvm.org/docs/AMDGPUOperandSyntax.html)
- [AMD Instinct MI300 (CDNA3) ISA reference guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf), [CDNA4 ISA reference guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), [all ISA documents](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)
- [rocprofv3 / ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
- [rocprof-trace-decoder](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder) ([binary releases](https://github.com/ROCm/rocprof-trace-decoder/releases)) and [ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer)
