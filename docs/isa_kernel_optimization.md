# ISA-Level Kernel Optimization with LLVM Tools

A guide to inspecting, analyzing, modifying, and recompiling AITER GPU kernel ISA using the ROCm LLVM toolchain.

## Overview

AITER ships optimized GPU kernels as compiled code objects (`.co` files). Sometimes you need to go deeper than source-level optimization. This guide shows how to:

1. Disassemble a `.co` kernel to human-readable ISA
2. Analyze instruction mix (MFMA, memory, LDS, DPP)
3. Extract a reassemblable `.s` file
4. Modify ISA instructions and recompile
5. Profile kernel performance with `rocprofv3`

All tools used are open-source ROCm components. No proprietary tools required.

## Prerequisites

- ROCm 6.x or later (tested on ROCm 7.2.1)
- LLVM tools: `llvm-objdump`, `clang++` (shipped with ROCm at `/opt/rocm/llvm/bin/`)
- `rocprofv3` (shipped with ROCm at `/opt/rocm/bin/`)
- Python 3.8+
- An AMD GPU (gfx90a, gfx942, or newer)

## Step 1: Locate the Kernel Object

AITER kernel `.co` files are typically found in the build directory or installed package:

```bash
# Find compiled kernel objects
find $(python -c "import aiter; print(aiter.__path__[0])") -name "*.co" | head -20

# Or look in the HSA directory
ls aiter/hsa/
```

For this guide, we'll use a Paged Attention kernel as an example:

```bash
KERNEL_CO="pa_bf16_pertokenFp8_gqa16_2tg_4w.co"
```

## Step 2: Disassemble to ISA

Use `llvm-objdump` to produce a full disassembly:

```bash
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942 $KERNEL_CO > kernel.isa
```

Replace `gfx942` with your target GPU architecture.

### Quick ISA Analysis

Count key instruction types to understand the kernel profile:

```bash
# Instruction statistics
echo "Total instructions: $(grep -cE '^\s+[0-9a-f]+:' kernel.isa)"
echo "MFMA (matrix) ops:  $(grep -c 'v_mfma_' kernel.isa)"
echo "Buffer loads:       $(grep -c 'buffer_load' kernel.isa)"
echo "LDS ops:            $(grep -c 'ds_' kernel.isa)"
echo "DPP ops:            $(grep -c '_dpp' kernel.isa)"
echo "Scalar ops:         $(grep -c '^[[:space:]]*s_' kernel.isa)"
```

### Read Kernel Metadata

```bash
# Extract register usage and resource requirements
/opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx942 -s -j .note $KERNEL_CO
```

Key metrics to look for:
- **SGPRs / VGPRs**: Register pressure limits occupancy
- **LDS size**: Shared memory per workgroup
- **Wavefront size**: 32 or 64

## Step 3: Extract Reassemblable Assembly

The raw `llvm-objdump` output is not directly reassemblable. A Python extraction script converts it to a valid `.s` file.

Create `extract_asm.py`:

```python
#!/usr/bin/env python3
"""Extract reassemblable .s from llvm-objdump -d output.

Handles branch label resolution using word-offset addressing:
  target_address = base_address + label_value * 4
"""
import re
import sys

def extract(isa_path, kernel_symbol, target="amdgcn-amd-amdhsa--gfx942"):
    with open(isa_path) as f:
        lines = f.readlines()

    # Find kernel section
    section_start = None
    for i, line in enumerate(lines):
        if f"<{kernel_symbol}>:" in line:
            section_start = i
            break
    if section_start is None:
        print(f"Kernel symbol '{kernel_symbol}' not found", file=sys.stderr)
        sys.exit(1)

    # Parse instruction address from first line
    first_instr_line = lines[section_start + 1].strip()
    base_addr = int(first_instr_line.split(":")[0].strip(), 16)

    # Collect instructions
    instructions = []
    for line in lines[section_start + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("Disassembly"):
            break
        m = re.match(r"\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]+\s+)+(.+)", stripped)
        if m:
            addr = int(m.group(1), 16)
            instr = m.group(2).strip()
            # Remove trailing hex comments
            instr = re.sub(r"\s*//\s*[0-9A-Fa-f]+$", "", instr)
            instructions.append((addr, instr))

    # Resolve branch labels (word offset: label_val * 4 + base_addr)
    branch_targets = {}
    for addr, instr in instructions:
        m = re.search(r"label_([0-9A-Fa-f]+)", instr)
        if m:
            label_val = int(m.group(1), 16)
            target_addr = base_addr + label_val * 4
            branch_targets[target_addr] = f"label_{m.group(1)}"

    # Emit .s file
    output = []
    output.append(f'  .amdgcn_target "{target}"')
    output.append(f"  .globl {kernel_symbol}")
    output.append(f"  .type {kernel_symbol}, @function")
    output.append(f"{kernel_symbol}:")

    for addr, instr in instructions:
        if addr in branch_targets:
            output.append(f"{branch_targets[addr]}:")
        output.append(f"  {instr}")

    output.append(f"  .size {kernel_symbol}, .-{kernel_symbol}")
    return "\n".join(output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <isa_file> <kernel_symbol> [target]")
        sys.exit(1)

    isa_file = sys.argv[1]
    symbol = sys.argv[2]
    target = sys.argv[3] if len(sys.argv) > 3 else "amdgcn-amd-amdhsa--gfx942"

    result = extract(isa_file, symbol, target)
    print(result)
```

Run the extraction:

```bash
# Find the kernel symbol name
grep "^[0-9a-f]" kernel.isa | head -1
# Example output: 0000000000001000 <_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE>:

# Extract reassemblable .s
python3 extract_asm.py kernel.isa \
    _ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4w > kernel_roundtrip.s
```

## Step 4: Recompile and Verify

Recompile the `.s` file back to a `.co`:

```bash
/opt/rocm/llvm/bin/clang++ \
    -x assembler \
    -target amdgcn-amd-amdhsa \
    -mcpu=gfx942 \
    -o kernel_recompiled.co \
    kernel_roundtrip.s
```

### Verify Binary Equivalence

Compare the `.text` section of the original and recompiled kernels:

```bash
# Extract .text sections
/opt/rocm/llvm/bin/llvm-objcopy -O binary -j .text $KERNEL_CO original_text.bin
/opt/rocm/llvm/bin/llvm-objcopy -O binary -j .text kernel_recompiled.co recompiled_text.bin

# Compare
md5sum original_text.bin recompiled_text.bin
diff <(xxd original_text.bin) <(xxd recompiled_text.bin) && echo "IDENTICAL" || echo "DIFFERS"
```

A successful round-trip produces identical `.text` sections. Metadata sections may differ (they are regenerated by the assembler), but the executable code is bit-exact.

### Producing a Loadable Kernel Object

The recompiled `.co` from Step 4 has a minimal `.note` section and may fail to load with `hipModuleLoad` ("no kernel image available"). The original `.co` contains rich AMDHSA metadata (kernel arguments, register counts, LDS size) that the HIP runtime requires.

To produce a loadable `.co`, inject the modified `.text` section back into the original kernel object:

```bash
# Copy original .co (preserves all metadata)
cp $KERNEL_CO kernel_modified.co

# Replace only the .text section with recompiled code
/opt/rocm/llvm/bin/llvm-objcopy --update-section .text=recompiled_text.bin kernel_modified.co
```

This preserves the original kernel descriptor, argument metadata, and ELF structure while swapping in the new executable code.

### Verifying Performance Equivalence

After swapping, benchmark both versions to confirm identical performance:

```bash
# Benchmark original
cp original_kernel.co $INSTALL_PATH/kernel.co
python benchmark.py  # record time

# Benchmark modified
cp kernel_modified.co $INSTALL_PATH/kernel.co
python benchmark.py  # compare time
```

On a PA decode kernel (bf16+fp8, GQA16, SEQ=4096), 3 runs of 500 iterations each showed original vs recompiled within ±3% noise — confirming zero performance regression from the round-trip.

## Step 5: Modify and Iterate

With a working round-trip established, you can now modify the `.s` file:

### Common ISA Optimizations

**Instruction scheduling** — Fill MFMA co-execution slots with independent operations:

```asm
; Before: MFMA followed by wait
v_mfma_f32_32x32x16_bf16 a[0:15], v[0:1], v[2:3], a[0:15]
s_nop 7        ; wasted cycles

; After: Fill with independent work
v_mfma_f32_32x32x16_bf16 a[0:15], v[0:1], v[2:3], a[0:15]
buffer_load_dwordx4 v[8:11], v4, s[0:3], 0  ; prefetch next tile
ds_read_b128 v[12:15], v5                     ; load from LDS
```

**Register pressure reduction** — Reuse registers to improve occupancy:

```asm
; Identify dead registers after their last use
; and reassign them for new values
```

**Memory access patterns** — Optimize buffer load/store coalescing and LDS bank conflicts.

After modifying, recompile and benchmark:

```bash
# Recompile modified kernel
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa \
    -mcpu=gfx942 -o kernel_modified.co kernel_modified.s

# Replace the .co in the AITER installation and re-run benchmark
```

## Step 6: Profile with rocprofv3

### Kernel-Level Tracing

Collect per-kernel dispatch timing with `--kernel-trace`:

```bash
rocprofv3 --kernel-trace -d ./profile_out -- python your_benchmark.py
```

This produces a SQLite database (`.db`) under the output directory with all kernel dispatches, including timestamps, grid dimensions, and kernel metadata.

### Output Formats

rocprofv3 supports multiple output formats:

```bash
# CSV (human-readable, easy to grep)
rocprofv3 --kernel-trace -f csv -d ./profile_out -- python your_benchmark.py

# JSON
rocprofv3 --kernel-trace -f json -d ./profile_out -- python your_benchmark.py

# Perfetto trace (open in https://ui.perfetto.dev)
rocprofv3 --kernel-trace -f pftrace -d ./profile_out -- python your_benchmark.py
```

### Querying the SQLite Database

The default output is a `.db` file. Table names include a UUID suffix. Use Python to query:

```python
import sqlite3, glob

db_path = glob.glob("profile_out/**/*results.db", recursive=True)[0]
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Find table names (they have UUID suffixes)
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%kernel_dispatch%'")
dispatch_table = c.fetchone()[0]

c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%kernel_symbol%'")
symbol_table = c.fetchone()[0]

# Top kernels by average duration
c.execute(f"""
    SELECT ks.kernel_name, COUNT(*) as cnt,
           AVG(d.end - d.start) as avg_ns,
           MIN(d.end - d.start) as min_ns,
           MAX(d.end - d.start) as max_ns
    FROM {dispatch_table} d
    JOIN {symbol_table} ks ON d.kernel_id = ks.id
    GROUP BY ks.kernel_name
    ORDER BY avg_ns DESC LIMIT 10
""")
print(f"{'Kernel':<60s} {'Count':>6s} {'Avg(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s}")
for name, cnt, avg, mn, mx in c.fetchall():
    print(f"  {name[:58]:<60s} {cnt:>6d} {avg/1000:>10.1f} {mn/1000:>10.1f} {mx/1000:>10.1f}")
```

### Filtering by Kernel Name

To focus on a specific kernel (e.g., paged attention):

```python
# Filter dispatches for PA kernels only
c.execute(f"""
    SELECT ks.kernel_name, COUNT(*) as cnt,
           AVG(d.end - d.start) as avg_ns,
           ks.arch_vgpr_count, ks.accum_vgpr_count,
           ks.sgpr_count, ks.group_segment_size
    FROM {dispatch_table} d
    JOIN {symbol_table} ks ON d.kernel_id = ks.id
    WHERE ks.kernel_name LIKE '%paged_attn%'
       OR ks.kernel_name LIKE '%pa_%'
    GROUP BY ks.kernel_name
    ORDER BY avg_ns DESC
""")
for name, cnt, avg, vgpr, agpr, sgpr, lds in c.fetchall():
    print(f"  {name[:70]}")
    print(f"    dispatches={cnt}, avg={avg/1000:.1f} us, "
          f"VGPR={vgpr}, AGPR={agpr}, SGPR={sgpr}, LDS={lds}")
```

### Combining Tracing Modes

Collect kernel, memory copy, and HIP runtime traces together for a full picture:

```bash
rocprofv3 --kernel-trace --memory-copy-trace --hip-trace \
    -d ./profile_out -- python your_benchmark.py
```

### Comparing Original vs Modified Kernel

After modifying the ISA and recompiling (Step 5), re-run the benchmark under `--kernel-trace` and compare:

```bash
# Profile original
rocprofv3 --kernel-trace -d ./profile_original -- python benchmark.py

# Swap in modified .co, profile again
rocprofv3 --kernel-trace -d ./profile_modified -- python benchmark.py

# Compare average kernel durations
```

### Advanced Thread Trace (ATT)

ATT captures instruction-level cycle counts, enabling precise bottleneck identification. It requires the `rocprof-trace-decoder` library, which is not shipped as a pre-built binary in ROCm 7.2.x but can be built from source:

```bash
# Build and install rocprof-trace-decoder
git clone --depth 1 --branch develop https://github.com/ROCm/rocm-systems.git
cd rocm-systems/projects/rocprof-trace-decoder
cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/rocm
cmake --build build -j$(nproc)
cmake --install build  # installs librocprof-trace-decoder.so to /opt/rocm/lib
```

Once installed, run ATT:

```bash
# Trace a single compute unit (CU 1)
rocprofv3 --att --att-target-cu 1 --kernel-trace \
    -d ./att_output -- python your_benchmark.py
```

This produces:
- `.att` files — raw per-wave binary trace data
- `.out` files — disassembled code objects for each kernel
- `results.db` — kernel dispatch database (same as `--kernel-trace`)

ATT options:

| Flag | Default | Description |
|------|---------|-------------|
| `--att-target-cu CU_ID` | 1 | Which compute unit (WGP) to trace |
| `--att-buffer-size BYTES` | 256MB | Trace buffer size per SE |
| `--att-shader-engine-mask MASK` | all | Bitmask of shader engines |
| `--att-gpu-index LIST` | all | Comma-separated GPU indices |

## Key Technical Details

### Branch Label Addressing

In `llvm-objdump` output, branch instructions reference labels like `label_0694`. These use **word offsets**, not byte offsets:

```
target_address = base_address + label_value * 4
```

For example, with `base_address = 0x1000` and `label_0694`:
```
target = 0x1000 + 0x694 * 4 = 0x1000 + 0x1A50 = 0x2A50
```

The extraction script handles this automatically.

### Architecture-Specific Considerations

| Architecture | GPU | Notes |
|-------------|-----|-------|
| gfx90a | MI210, MI250 | CDNA2, wavefront 64 |
| gfx942 | MI300X | CDNA3, wavefront 64, MFMA co-execution |

This workflow applies to all AMDGPU architectures supported by LLVM. Adjust the `--mcpu` flag and `.amdgcn_target` string to match your target GPU.

### Typical Kernel Instruction Profile

For a well-optimized attention kernel on gfx942:

| Instruction Category | Count | Purpose |
|---------------------|-------|---------|
| MFMA (v_mfma_*) | ~192 | Matrix multiply-accumulate |
| Buffer loads | ~100 | Global memory reads |
| LDS ops (ds_*) | ~300+ | Shared memory access |
| DPP ops (*_dpp) | ~300+ | Cross-lane data movement |
| Scalar ops (s_*) | ~200+ | Control flow, address calculation |

## Troubleshooting

**llvm-objdump not found**: Use the full path `/opt/rocm/llvm/bin/llvm-objdump`.

**Recompiled .text differs from original**: Check that branch labels are resolved correctly. The extraction script must use word-offset addressing (multiply label value by 4).

**ATT trace fails with "trace-decoder not found"**: Build and install `librocprof-trace-decoder.so` from source (see the ATT section above). The library is not included in ROCm 7.2.x binary packages.

**Metadata sections differ after round-trip**: This is expected. The `.text` (executable code) section should be identical. Metadata is regenerated by the assembler and may have different formatting.

## References

- [LLVM AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMDGPU ISA Reference (GFX9)](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX9.html)
- [rocprofv3 Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)
- [rocprof-trace-decoder](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder) — ATT trace decoder library (build from source)
