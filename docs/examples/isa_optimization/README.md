# ISA Kernel Optimization Examples

Code examples for the [ISA-Level Kernel Optimization Guide](../../isa_kernel_optimization.md).

## Quick Start

### Using Docker (recommended)

```bash
docker build -t aiter-isa-opt .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    aiter-isa-opt
```

Inside the container:

```bash
# Full round-trip on a PA kernel
./roundtrip.sh /path/to/kernel.co --mcpu gfx942

# Analyze instruction mix of a .co file
python3 analyze_kernel.py isa /path/to/kernel.co --mcpu gfx942

# Profile with rocprofv3 and analyze results
rocprofv3 --kernel-trace -d ./profile_out -- python3 your_benchmark.py
python3 analyze_kernel.py profile ./profile_out --filter "pa_"
```

### Without Docker

Requires ROCm 6.x+ installed with LLVM tools and rocprofv3.

```bash
# Run the round-trip script directly
./roundtrip.sh kernel.co --mcpu gfx942

# Or use the individual scripts
python3 extract_asm.py kernel.isa SYMBOL_NAME --target amdgcn-amd-amdhsa--gfx942 -o kernel.s
python3 analyze_kernel.py isa kernel.co
```

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_asm.py` | Extract reassemblable `.s` from `llvm-objdump -d` output |
| `analyze_kernel.py` | ISA instruction mix analysis and rocprofv3 profile parsing |
| `roundtrip.sh` | End-to-end round-trip: disassemble, extract, recompile, verify |
| `Dockerfile` | Development environment with all tools pre-installed |

## Workflow

```
kernel.co
  │
  ├─ llvm-objdump -d ──► kernel.isa
  │                         │
  │                  extract_asm.py ──► kernel.s
  │                                      │
  │                              (edit ISA here)
  │                                      │
  │                           clang++ -x assembler ──► recompiled.co
  │                                                      │
  │              llvm-objcopy -O binary -j .text ──► recompiled_text.bin
  │                                                      │
  └─── cp ──► modified.co ◄── llvm-objcopy --update-section .text=recompiled_text.bin
                  │
           (loadable, with original metadata preserved)
```
