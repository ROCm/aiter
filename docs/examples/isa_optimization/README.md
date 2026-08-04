# ISA Kernel Optimization Examples

Scripts for the [ISA-Level Kernel Optimization Guide](../../isa_kernel_optimization.md).

| File | Purpose |
|------|---------|
| `extract_asm.py` | Rebuild a self-contained, reassemblable `.s` (instructions + kernel descriptor + metadata) from a `.co` |
| `roundtrip.sh` | `.co` → `.s` → reassemble → verify `.text`, kernel descriptor, metadata and ELF flags against the original |
| `analyze_kernel.py` | Instruction mix / register / LDS summary of a `.co`, and per-kernel timing from a `rocprofv3 --kernel-trace` database |
| `Dockerfile` | ROCm LLVM tools + rocprofv3 + ATT trace decoder |

## Quick start

The scripts only need the LLVM tools shipped with ROCm (`$ROCM_PATH/llvm/bin`,
default `/opt/rocm/llvm/bin`) and Python 3.

```bash
# Round-trip one of the shipped paged-attention kernels (arch is read from the ELF)
./roundtrip.sh ../../../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co --keep ./work

# What is in a .co: kernels, branch labels, register/LDS usage, instruction mix
python3 extract_asm.py   ../../../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co --list
python3 analyze_kernel.py isa ../../../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co

# Edit ./work/kernel.s, then rebuild a loadable .co over the shipped one
# (a develop install loads kernels from the checkout's hsa/ tree; git keeps the original)
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 \
    -o ../../../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co ./work/kernel.s

# Test it -- the "[aiter] LoadKernel: ... hsaco: <path>" log line shows which file was loaded
python3 ../../../op_tests/test_pa.py
git checkout -- ../../../hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co   # restore

# Profile
rocprofv3 --kernel-trace --stats -d ./profile_out -- python3 ../../../op_tests/test_pa.py
python3 analyze_kernel.py profile ./profile_out --filter pa_
```

Note that `import aiter` sets `AITER_ASM_DIR` itself (to `<AITER_META_DIR>/hsa/`),
so exporting it in the shell does not redirect kernel loading; see the guide.

### Using Docker

```bash
docker build -t aiter-isa-opt .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /path/to/aiter:/aiter aiter-isa-opt
./roundtrip.sh /aiter/hsa/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co
```

## Workflow

```
kernel.co ──┬─ llvm-objdump -d            (instructions; labels come from .symtab)
            ├─ llvm-objdump -D -j .rodata (kernel descriptor → .amdhsa_* directives)
            └─ llvm-readelf --notes       (metadata → .amdgpu_metadata YAML)
                     │
              extract_asm.py ──► kernel.s   (edit here: instructions *and* resources)
                     │
   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfxNNN ──► kernel_rebuilt.co
                     │
   verify (roundtrip.sh) ─► test (op_tests/) ─► profile (rocprofv3)
```

`roundtrip.sh` exits non-zero if the rebuilt object is not byte-identical to the
input; see "Known re-encoding differences" in the guide for the cases where that
is expected.
